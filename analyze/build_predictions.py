"""對 test.csv 做一次全量推論，把結果快取成 predictions.parquet。

背景：test.csv 有 2.76GB，不能整份讀進記憶體，只能分塊（chunk）處理。但
client.py/server.py 的 preprocess_data() 用 pd.Categorical(col).codes 幫非數值欄位
編碼，這個編碼是「看當下這批資料有哪些類別」臨時決定的——同一個字串在不同 chunk
會被分到不同整數，直接照搬 preprocess_data() 逐塊處理會讓預測失效。

解法分兩階段：
  Phase 1：先掃過全檔（只讀候選的非數值欄位），把每一欄「完整」的類別集合蒐集起來，
           排序後固定成一份全域 category -> code 對照表，存成 category_map.json。
  Phase 2：再分塊讀「全部」欄位，套用跟 client.py 完全一樣的前處理，唯一差異是
           categorical 編碼改用 Phase 1 算好的固定對照表（pd.Categorical(col,
           categories=固定清單)），保證同一個字串無論落在哪個 chunk 都得到同一個整數。

哪些欄位需要固定對照表，用 test.csv 前 1000 列的 dtype 做初篩（見
detect_candidate_columns()）。這個初篩可能漏掉「前 1000 列剛好沒缺值、但後面某個
chunk 出現缺值或字串」的欄位——因為 convert_type() 對缺值（NaN）也會回傳空字串 ''，
一缺值該欄就變成非數值欄，這在這份資料裡是常態而不是例外（Wireshark 攤平出來的欄位
大多只在特定封包類型才有值）。Phase 2 因此對「初篩判定為數值欄」的欄位加一道即時檢查：
若某個 chunk 讓這欄變成非數值，直接中止並報錯，而不是安靜地產生錯的編碼。

用法：
  python build_predictions.py --nrows 200000   # 小規模驗證
  python build_predictions.py                  # 全量（不指定 --nrows）
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # 舊版 Python 或非一般 stream（例如某些 pipe）沒有 reconfigure，忽略即可

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
TEST_CSV = BASE_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "global_model_latest.ubj"
RESULTS_DIR = BASE_DIR / "results"
CATEGORY_MAP_PATH = RESULTS_DIR / "category_map.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.parquet"

# 逐字複製自 our-project/model_training/client.py:60-64（server.py 內容相同）。
# 這裡照抄而不是 import，是因為 client.py 是 our-project/ 底下的正式程式碼、
# 不能動它去 export 這個常數；照抄的代價是兩邊要手動保持同步，若日後
# client.py 改了這個 mapping，這裡要跟著改。
ATTACK_MAPPING = {
    "observe": 0,
    "metasploit SYN flood": 1,
    "nmap discovery": 1,
    "nmap SYN flood": 1,
    "ros2 node crashing": 1,
    "ros2 reconnaissance": 1,
    "ros2 reflection": 1,
}

PROBE_NROWS = 2000  # 拿來估計「每列大約占多少記憶體」的樣本大小
CANDIDATE_SCREEN_NROWS = 1000  # 判斷「哪些欄位可能是非數值欄」的樣本大小
MEMORY_SAFETY_FACTOR = 5  # chunksize 估算的保守係數，理由見 compute_chunksize()


def convert_type(x):
    """逐字複製自 client.py:30-44（server.py:13-29 相同）。

    決定一個 CSV 儲存格最終要當成數字還是字串：已經是數字（非 bool）就原樣回傳；
    缺值或空字串一律變成空字串 ''；其餘先試著轉成數字，轉不了才當字串留著。
    """
    if (isinstance(x, (int, float, np.number)) and not pd.isna(x)) and not isinstance(x, bool):
        return x

    if pd.isna(x) or x == "":
        return ""

    try:
        return pd.to_numeric(x)
    except Exception:
        try:
            return str(x)
        except Exception:
            return ""


def to_native(x):
    """把 numpy 純量（np.int64、np.float64...）轉成對應的 Python 內建型別。

    category_map.json 要能被 json.dump 直接序列化，numpy 純量不行；轉成 Python
    內建型別後，跟從 JSON 讀回來的值比較時型別、hash 也才會一致，不會出現
    np.float64(673.0) 和 json 讀回來的 float(673.0) 被當成不同類別的問題。
    """
    if isinstance(x, np.generic):
        return x.item()
    return x


def convert_cell(x):
    """convert_type() 接 to_native()，Phase 1 蒐集類別、Phase 2 套用編碼都用這個。

    兩階段一定要用同一個函式，「同一個原始字串在兩階段被轉成同一個 Python 值」
    這件事才能保證成立——這正是整支腳本要解決的一致性問題的核心。
    """
    return to_native(convert_type(x))


def detect_candidate_columns():
    """讀 test.csv 前 CANDIDATE_SCREEN_NROWS 列，抓出「可能需要類別編碼」的欄位。

    做法跟 preprocess_data() 一致：對每個非 attack 欄位套用 convert_cell()，再用
    select_dtypes 排除數值/布林欄位，剩下的就是候選欄位。回傳 (候選欄位清單,
    完整原始欄位清單)。
    """
    print(f"[Phase 1a] 讀取前 {CANDIDATE_SCREEN_NROWS} 列判斷欄位型別...")
    df = pd.read_csv(TEST_CSV, nrows=CANDIDATE_SCREEN_NROWS, low_memory=False)
    all_columns = df.columns.tolist()
    feature_cols = [c for c in all_columns if c != "attack"]

    df[feature_cols] = df[feature_cols].map(convert_cell)
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    print(f"[Phase 1a] 全部 {len(feature_cols)} 個特徵欄位中，{len(non_numeric)} 個判定為候選類別欄位。")
    return non_numeric, all_columns


def build_category_map(candidate_columns, nrows, chunksize):
    """Phase 1b：分塊掃過（最多 nrows 列）全檔，蒐集每個候選欄位的完整類別集合。

    每欄的類別依「先數值後字串，各自排序」的規則排成固定順序清單，清單的索引
    值就是之後 Phase 2 要用的整數編碼。這個順序是這支腳本自訂的，不等於
    client.py 訓練當下 pd.Categorical 用過的編碼——那組編碼早就跟著各個 client
    當時看到的資料一起消失了，找不回來。這支腳本能保證的只有「test.csv 內部
    自己一致」，不保證跟訓練時的類別編碼對得上；這是類別欄位在這個架構下
    本來就有的限制，不是這支腳本能解的問題。
    """
    print(f"[Phase 1b] 分塊掃描全部候選欄位的類別集合（共 {len(candidate_columns)} 欄）...")
    unique_values = {col: set() for col in candidate_columns}

    reader = pd.read_csv(
        TEST_CSV, usecols=candidate_columns, chunksize=chunksize, low_memory=False, nrows=nrows
    )
    total_rows = 0
    start = time.time()
    for i, chunk in enumerate(reader):
        chunk = chunk.map(convert_cell)
        for col in candidate_columns:
            unique_values[col].update(chunk[col].unique().tolist())
        total_rows += len(chunk)
        elapsed = time.time() - start
        print(f"[Phase 1b] 第 {i + 1} 塊掃描完成，累計 {total_rows} 列，耗時 {elapsed:.1f}s")

    def sort_key(v):
        if isinstance(v, (int, float)):
            return (0, float(v))
        return (1, str(v))

    category_map = {}
    for col, values in unique_values.items():
        sorted_values = sorted(values, key=sort_key)
        category_map[col] = sorted_values
        if len(sorted_values) > 1000:
            print(f"[警告] 欄位 {col!r} 有 {len(sorted_values)} 種類別，數量偏多，確認一下是不是預期內。")

    return category_map, total_rows


def save_category_map(category_map, nrows, screen_nrows, scanned_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "source_file": str(TEST_CSV),
            "candidate_screen_nrows": screen_nrows,
            "built_from_nrows": nrows,
            "scanned_rows": scanned_rows,
            "num_candidate_columns": len(category_map),
            "note": (
                "此對照表只在建立當下所讀的資料範圍內保證正確（見 built_from_nrows / "
                "scanned_rows）。用 --nrows 跑小規模驗證時，這份對照表只涵蓋前 nrows 列，"
                "全量執行時會被完全重建，兩者不能混用。"
            ),
        },
        "columns": category_map,
    }
    with open(CATEGORY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[Phase 1b] 類別對照表已存到 {CATEGORY_MAP_PATH}")


def preprocess_chunk(df, candidate_columns, category_map, feature_names):
    """對一個 chunk 套用跟 preprocess_data()（client.py:47-76）等價的前處理。

    順序刻意跟原函式一致：先轉型別 → 處理 inf/NaN → 處理 attack 欄 → 丟掉
    Unnamed/timestamp 欄 → 類別欄位編碼 → 欄名清理 → 轉 float32。跟原函式的
    唯一差異在類別欄位編碼那一步：原函式用 pd.Categorical(col).codes（依當下
    這批資料現有的類別），這裡改用 pd.Categorical(col, categories=固定清單)
    （Phase 1 掃全檔算出來的固定清單），讓同一個字串在任何 chunk 都編到同一個
    整數。

    回傳 (features_df, y_true, attack_raw)。
    """
    attack_raw = df["attack"].copy()
    feature_cols = [c for c in df.columns if c != "attack"]

    df[feature_cols] = df[feature_cols].map(convert_cell)
    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    if "attack" not in df.columns:
        raise RuntimeError(
            "這個 chunk 處理完 attack 欄不見了（dropna(thresh=1) 把它丟掉了，代表整個 "
            "chunk 的 attack 欄全是缺值）。這不是預期內的資料狀態，中止執行。"
        )
    df["attack"] = df["attack"].replace(ATTACK_MAPPING).infer_objects(copy=False)
    y_true = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)

    df = df.drop(
        columns=[c for c in df.columns if "Unnamed" in c or "timestamp" in c], errors="ignore"
    )

    for col in candidate_columns:
        if col not in df.columns:
            continue  # 被上面 Unnamed/timestamp 那行丟掉了，跳過
        df[col] = pd.Categorical(df[col], categories=category_map[col]).codes

    # 安全網：初篩判定為「數值欄」的欄位，理論上這個 chunk 也該是數值欄；
    # 如果不是，代表 Phase 1 用前 1000 列做的初篩漏掉了這一欄（比如這欄剛好
    # 在前 1000 列都沒缺值，但這個 chunk 出現缺值或字串），若真的發生就直接
    # 中止，而不是讓 pd.Categorical 完全沒跑到、留下字串混進 float32 陣列裡。
    remaining_non_numeric = df.drop(columns=["attack"]).select_dtypes(
        exclude=[np.number, "bool"]
    ).columns.tolist()
    if remaining_non_numeric:
        raise RuntimeError(
            f"以下欄位在這個 chunk 變成非數值，但 Phase 1 初篩沒有把它們列為候選類別欄位："
            f"{remaining_non_numeric}。代表前 {CANDIDATE_SCREEN_NROWS} 列的初篩樣本不夠代表性，"
            "需要把這些欄位加進候選清單重跑，不能忽略。"
        )

    df.columns = [str(c).replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_") for c in df.columns]
    df = df.astype(np.float32)

    features = df.drop(columns=["attack"])
    if list(features.columns) != feature_names:
        missing = set(feature_names) - set(features.columns)
        extra = set(features.columns) - set(feature_names)
        raise RuntimeError(
            f"這個 chunk 前處理完的欄位跟模型的 feature_names 對不上。"
            f"模型缺少但這裡有的欄位：{sorted(extra)[:10]}；"
            f"模型需要但這裡沒有的欄位：{sorted(missing)[:10]}。"
        )

    return features, y_true, attack_raw


def compute_chunksize(memory_budget_bytes, override=None):
    """量測 test.csv 每列大約占多少記憶體，反推一個安全的 chunksize。

    量測方式：實際讀 PROBE_NROWS 列，用 memory_usage(deep=True) 抓 pandas 真正
    配置的記憶體（deep=True 才會把 object 欄位裡的字串內容也算進去，不然像這種
    有 400 多個欄位、其中大半是字串/混合型別的資料會嚴重低估）。除以
    MEMORY_SAFETY_FACTOR 是因為前處理過程中同時會有：原始 chunk、.map()
    轉型後的新陣列、轉 float32 後的 features 陣列、DMatrix 內部的另一份拷貝，
    尖峰記憶體是「單純讀進來的大小」的好幾倍，不是隨便抓的數字，沒有實際量測
    這個放大倍率，用保守係數換取安全邊際。
    """
    if override is not None:
        print(f"[Info] chunksize 使用手動指定值：{override}")
        return override

    print(f"[Info] 量測記憶體佔用中（讀 {PROBE_NROWS} 列）...")
    probe = pd.read_csv(TEST_CSV, nrows=PROBE_NROWS, low_memory=False)
    bytes_per_row = probe.memory_usage(deep=True).sum() / len(probe)

    chunksize = int(memory_budget_bytes / bytes_per_row / MEMORY_SAFETY_FACTOR)
    chunksize = max(5000, min(chunksize, 300_000))

    print(
        f"[Info] 每列約 {bytes_per_row:.0f} bytes（deep memory_usage），"
        f"記憶體上限 {memory_budget_bytes / 1024**3:.1f}GB，"
        f"保守係數 {MEMORY_SAFETY_FACTOR}x -> chunksize = {chunksize}"
    )
    return chunksize


def estimate_total_rows(nrows):
    """粗估 test.csv 總列數，只用來顯示進度，不追求精確。

    不跑 wc -l（對 2.76GB 檔案來說不算貴，但沒必要），改用「檔案大小 / 平均每列
    位元組數」估計，平均值來自前面 PROBE_NROWS 列的實際位元組長度。這份資料
    的列長度會因為 TCP payload、SSL 欄位是否有值而有落差，估出來的塊數僅供
    進度顯示參考，不是精確值。
    """
    if nrows is not None:
        return nrows

    with open(TEST_CSV, "rb") as f:
        sample = f.read(5 * 1024 * 1024)
    newline_count = sample.count(b"\n")
    if newline_count == 0:
        return None
    avg_row_bytes = len(sample) / newline_count
    file_size = TEST_CSV.stat().st_size
    return int(file_size / avg_row_bytes)


def run_inference(candidate_columns, category_map, chunksize, nrows, feature_names, booster):
    print(f"[Phase 2] 開始分塊推論（chunksize={chunksize}, nrows={nrows or '全量'}）...")
    estimated_total_rows = estimate_total_rows(nrows)
    estimated_total_chunks = (
        max(1, -(-estimated_total_rows // chunksize)) if estimated_total_rows else None
    )

    reader = pd.read_csv(TEST_CSV, chunksize=chunksize, low_memory=False, nrows=nrows)
    result_frames = []
    total_rows = 0
    start = time.time()

    for i, chunk in enumerate(reader):
        features, y_true, attack_raw = preprocess_chunk(
            chunk, candidate_columns, category_map, feature_names
        )
        dmatrix = xgb.DMatrix(features.values, feature_names=feature_names)
        y_score = booster.predict(dmatrix)

        result_frames.append(
            pd.DataFrame(
                {
                    "attack_raw": attack_raw.values,
                    "y_true": y_true.values,
                    "y_score": y_score,
                }
            )
        )
        total_rows += len(chunk)
        elapsed = time.time() - start
        total_label = f"~{estimated_total_chunks}" if estimated_total_chunks else "?"
        print(
            f"[Phase 2] 第 {i + 1}/{total_label} 塊完成，累計 {total_rows} 列，耗時 {elapsed:.1f}s"
        )

    return pd.concat(result_frames, ignore_index=True)


def print_summary(predictions):
    print("\n===== 推論結果摘要 =====")
    print(f"總筆數: {len(predictions)}")

    print("\ny_true 分佈:")
    print(predictions["y_true"].value_counts())

    y_score = predictions["y_score"]
    print(f"\ny_score: min={y_score.min():.6f}, max={y_score.max():.6f}, mean={y_score.mean():.6f}")

    y_true = predictions["y_true"].values
    y_pred = (y_score.values > 0.5).astype(int)

    print("\n整體指標（threshold=0.5）:")
    print(f"  accuracy  = {accuracy_score(y_true, y_pred):.4f}")
    print(f"  precision = {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  recall    = {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  f1        = {f1_score(y_true, y_pred, zero_division=0):.4f}")

    if len(set(y_true)) < 2:
        print(
            "  AUC       = 無法計算（y_true 只有一個類別 "
            f"{set(y_true)}，這個資料範圍內沒有正負樣本對照，不是程式錯誤）"
        )
    else:
        print(f"  AUC       = {roc_auc_score(y_true, y_score):.4f}")


def main():
    parser = argparse.ArgumentParser(description="對 test.csv 做全量推論並快取成 parquet")
    parser.add_argument("--nrows", type=int, default=None, help="限制讀取列數（小規模驗證用，例如 200000）；不指定則跑全量")
    parser.add_argument("--chunksize", type=int, default=None, help="覆寫自動計算出的 chunksize")
    parser.add_argument("--memory-budget-gb", type=float, default=4.0, help="用來自動估算 chunksize 的記憶體上限（GB）")
    args = parser.parse_args()

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"找不到 {TEST_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 {MODEL_PATH}")

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    feature_names = booster.feature_names

    chunksize = compute_chunksize(args.memory_budget_gb * 1024**3, override=args.chunksize)

    candidate_columns, _ = detect_candidate_columns()
    category_map, scanned_rows = build_category_map(candidate_columns, args.nrows, chunksize)
    save_category_map(category_map, args.nrows, CANDIDATE_SCREEN_NROWS, scanned_rows)

    predictions = run_inference(
        candidate_columns, category_map, chunksize, args.nrows, feature_names, booster
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"\n[Info] 預測結果已存到 {PREDICTIONS_PATH}")

    print_summary(predictions)


if __name__ == "__main__":
    main()
