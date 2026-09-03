"""跑 ERR/LFR 四格實驗矩陣：{無防禦, ERR/LFR} x {無攻擊, 有攻擊}，全部用
bagging，比較有無 ERR/LFR 防禦、在有無攻擊下的表現。「無防禦」兩格直接
沿用先前已跑過的模型不重新訓練，「ERR/LFR」兩格是本檔案新跑的。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 要改新跑哪幾組 ERR/LFR 設定，改 FRESH_ERR_LFR_CONFIGS；REUSED_NO_DEFENSE_MODELS/
# REUSED_SOURCE_LABEL 是沿用任務 13 既有模型的對照表，其餘是共用路徑與訓練固定參數。
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd
import xgboost as xgb

import server as srv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
NUM_CLIENTS = 5
NUM_ROUNDS = 10
SERVER_ADDRESS = "127.0.0.1:8085"
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_data", "chunk_6.csv")
TEST_NATURAL_PATH = os.path.join(BASE_DIR, "split_data", "test_natural.csv")
TEST_RARE_PATH = os.path.join(BASE_DIR, "split_data", "test_rare.csv")
LEAF_SCALE = 0.5
NUM_TO_EXCLUDE = 1

# 任務 13 留下的「無防禦」round-10 模型，兩格都直接沿用（見上方 docstring）
REUSED_NO_DEFENSE_MODELS = {
    "no_attack_no_defense": os.path.join(BASE_DIR, "outputs", "models", "attack_no_attack", f"global_model_round_{NUM_ROUNDS}.ubj"),
    "attack_no_defense": os.path.join(BASE_DIR, "outputs", "models", "attack_attack_c1_full", f"global_model_round_{NUM_ROUNDS}.ubj"),
}
REUSED_SOURCE_LABEL = {
    "no_attack_no_defense": "no_attack",
    "attack_no_defense": "attack_c1_full",
}

# 本檔案要新跑的兩格：(cell_label, malicious_client_ids, label_flip_rate)
FRESH_ERR_LFR_CONFIGS = [
    ("no_attack_err_lfr", set(), 1.0),
    ("attack_err_lfr", {1}, 1.0),
]

# ==================== 區塊 2：log 解析規則 ====================
# 綁死 server.py 在 ERR/LFR 模式下印出來的 log 文字格式，用來抓逐輪指標、參與節點數、排除名單。
ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) ERR/LFR-filtered bagging kept (\d+)/\d+ client models "
    r"\(excluded=\[(.*?)\]\) \(accuracy=([\d.]+), f1=([\d.]+), margin=\[(-?[\d.]+), (-?[\d.]+)\], "
    r"margin_mean=(-?[\d.]+)\)"
)
PARTICIPATION_RE = re.compile(r"\[ErrLfr\] Round (\d+): (\d+)/(\d+) client\(s\) participated")
EXCLUDED_RE = re.compile(
    r"\[ErrLfr\] Round (\d+) excluded client_id\(s\)=\[(.*?)\] "
    r"\(ERR flagged idx=\[(.*?)\], LFR flagged idx=\[(.*?)\], union size=(\d+)/(\d+)\)"
)


# ==================== 區塊 3：子行程收尾工具 ====================
# 每組實驗跑完（或逾時）之後，用來把 server/client 子行程都關掉的共用函式。
def stop_process(proc):
    """終止子行程，先 terminate，5 秒沒關掉就 kill。"""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_one_err_lfr(label, malicious_ids, flip_rate, summary_rows, participation_rows, exclusion_rows):
    """跑一組 ERR/LFR 設定的完整 10 輪聯邦訓練，解析伺服器 log 取得逐輪
    accuracy/F1（chunk_6）、參與節點數、每輪 ERR/LFR 排除了哪些 client_id。

    輸入：label 是這組的檔名/資料夾標籤；malicious_ids 是要標記成攻擊節點
    的 client_id 集合；flip_rate 是這些節點的標籤翻轉比例；
    summary_rows/participation_rows/exclusion_rows 是呼叫端傳入的 list，
    這個函式直接把結果 append 進去（就地修改，無回傳值）。
    """
    # ==================== 區塊 4：啟動 server ====================
    print(f"\n===== {label} (malicious={sorted(malicious_ids)}, flip_rate={flip_rate}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, "outputs", "models", f"err_lfr_{label}")
    log_dir = os.path.join(BASE_DIR, "outputs", "logs", f"err_lfr_{label}")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    server_log_path = os.path.join(log_dir, "server.log")
    server_log = open(server_log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [
            sys.executable, os.path.join(BASE_DIR, "server.py"),
            f"--model_dir={model_dir}",
            f"--num_clients={NUM_CLIENTS}",
            f"--num_rounds={NUM_ROUNDS}",
            f"--server_address={SERVER_ADDRESS}",
            f"--validation_data_path={VAL_DATA_PATH}",
            "--aggregation=err_lfr",
            f"--leaf_scale={LEAF_SCALE}",
            f"--num_to_exclude={NUM_TO_EXCLUDE}",
        ],
        stdout=server_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
    )
    time.sleep(3)

    # ==================== 區塊 5：啟動 client ====================
    client_procs = []
    client_logs = []
    for i in range(1, NUM_CLIENTS + 1):
        data_path = os.path.join(BASE_DIR, "split_data", f"chunk_{i}.csv")
        client_log = open(os.path.join(log_dir, f"client_{i}.log"), "w", encoding="utf-8")
        client_logs.append(client_log)
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "client.py"),
            f"--client_id={i}",
            f"--data_path={data_path}",
            f"--server_address={SERVER_ADDRESS}",
            "--aggregation=bagging",
        ]
        if i in malicious_ids:
            cmd += ["--malicious", f"--label_flip_rate={flip_rate}"]
        proc = subprocess.Popen(cmd, stdout=client_log, stderr=subprocess.STDOUT, cwd=BASE_DIR)
        client_procs.append(proc)

    # ==================== 區塊 6：等待與收拾 ====================
    start = time.time()
    timeout_s = 900  # 比任務 13 的 600s 寬鬆——ERR/LFR 每輪多做 N 次候選評估，更慢
    while server_proc.poll() is None:
        if time.time() - start > timeout_s:
            print(f"[Warning] {label} server did not finish within {timeout_s}s, killing.")
            break
        time.sleep(1)
    elapsed = time.time() - start
    print(f"[Info] server exited after {elapsed:.1f}s (exit code {server_proc.poll()})")

    stop_process(server_proc)
    for proc in client_procs:
        stop_process(proc)
    server_log.close()
    for cl in client_logs:
        cl.close()

    # ==================== 區塊 7：解析 log ====================
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    for m in ROUND_RE.finditer(server_text):
        rnd, n_kept, excluded_str, acc, f1, mmin, mmax, mmean = m.groups()
        summary_rows.append({
            "config": label, "round": int(rnd),
            "accuracy": float(acc), "f1": float(f1),
            "margin_min": float(mmin), "margin_max": float(mmax), "margin_mean": float(mmean),
        })
    for m in PARTICIPATION_RE.finditer(server_text):
        rnd, n_part, n_total = m.groups()
        participation_rows.append({
            "config": label, "round": int(rnd),
            "participating_clients": int(n_part), "num_clients": int(n_total),
        })
    for m in EXCLUDED_RE.finditer(server_text):
        rnd, excluded_ids, err_idx, lfr_idx, union_n, total_n = m.groups()
        exclusion_rows.append({
            "config": label, "round": int(rnd),
            "excluded_client_ids": excluded_ids,
            "err_flagged_idx": err_idx, "lfr_flagged_idx": lfr_idx,
            "union_size": int(union_n), "num_clients": int(total_n),
        })


# ==================== 區塊 8：逐攻擊類型 recall 計算 ====================
def per_type_recall(bst, df_raw):
    """對同一個模型，用原始 attack 字串欄位分組算每種類型「被判定為攻擊」
    的比例，回傳 [(攻擊類型, 該類型筆數, 比例, "recall"或"FPR"), ...]。
    跟 eval_baseline_on_test_natural.py 的同名函式邏輯相同（此處獨立一份
    是因為要對 test_rare.csv 而非 test_natural.csv 呼叫）。
    """
    attack_labels = df_raw["attack"].copy()
    df = srv.preprocess_data(df_raw.copy())
    X = df.iloc[:, :-1]
    dm = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
    preds = (bst.predict(dm) > 0.5).astype(int)

    rows = []
    for atype in attack_labels.unique():
        mask = (attack_labels == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        rate = float(preds[mask].mean())
        metric = "FPR" if atype == "observe" else "recall"
        rows.append((atype, n, rate, metric))
    return rows


# ==================== 區塊 9：最終報告彙整 ====================
def build_final_report(cell_model_paths, final_rows):
    """對四格各自的 round-10 模型，讀一次 test_natural 算整體指標、讀一次
    test_rare 算逐攻擊類型 recall/FPR，寫進 results/err_lfr_final_report.csv
    與逐格的 results/err_lfr_final_recall_<cell>.txt。

    輸入：cell_model_paths 是 {cell_label: model_path} 的 dict；final_rows
    是呼叫端傳入的 list，這個函式把四格的整體指標 append 進去。
    決策路徑（ERR/LFR 排除判斷）已經在 run_one_err_lfr() 裡跑完、只讀了
    chunk_6，這裡是訓練結束後才進行的最終報告階段，讀 TEST_DATA 沒有問題。
    """
    with tempfile.TemporaryDirectory() as scratch_model_dir:
        strategy = srv.XGBoostStrategy(
            model_dir=scratch_model_dir,
            num_clients=NUM_CLIENTS,
            val_data_path=VAL_DATA_PATH,
            test_natural_data_path=TEST_NATURAL_PATH,
        )
        df_test_rare_raw = pd.read_csv(TEST_RARE_PATH)

        for cell, model_path in cell_model_paths.items():
            if not os.path.exists(model_path):
                print(f"[Warning] {cell}: model not found at {model_path}, skipping final report.")
                continue
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            overall = strategy._evaluate_model_on_server(model_bytes, dataset="test_natural")
            final_rows.append({
                "cell": cell, "model_path": model_path,
                "test_natural_accuracy": overall["accuracy"],
                "test_natural_precision": overall["precision"],
                "test_natural_recall": overall["recall"],
                "test_natural_f1": overall["f1"],
            })

            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            rows = per_type_recall(bst, df_test_rare_raw)
            recall_path = os.path.join(RESULTS_DIR, f"err_lfr_final_recall_{cell}.txt")
            with open(recall_path, "w", encoding="utf-8") as f:
                f.write(f"model={model_path}\n")
                f.write(
                    f"# 這份檔案的逐攻擊類型 recall/FPR 算在 test_rare.csv 上；\n"
                    f"# 整體 accuracy/precision/recall/f1 是算在 test_natural.csv 上（見\n"
                    f"# err_lfr_final_report.csv），兩者不是同一份資料，不能直接放在同一列比。\n"
                    f"test_natural overall: accuracy={overall['accuracy']:.4f} "
                    f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
                    f"f1={overall['f1']:.4f}\n\n"
                )
                f.write(f"{'attack_type':30s} {'n':>8s} {'value':>10s}  metric\n")
                for atype, n, rate, metric in rows:
                    f.write(f"{atype:30s} {n:8d} {rate:10.4f}  {metric}\n")
            print(f"[Info] wrote {recall_path}")


# ==================== 區塊 10：主流程／寫出結果 ====================
def main():
    """1) 沿用任務 13 的兩個「無防禦」round-10 模型；2) 新跑兩組 ERR/LFR
    設定的完整 10 輪訓練；3) 對四格的 round-10 模型統一跑 test_natural/
    test_rare 最終報告評估；4) 把逐輪 summary/participation/exclusion 以及
    最終報告寫成 results/ 底下的 csv/txt。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows, participation_rows, exclusion_rows = [], [], []

    # --- 沿用任務 13 的逐輪資料（從已寫好的 csv 讀回，換上四格用的 label） ---
    old_summary_path = os.path.join(RESULTS_DIR, "attack_injection_summary.csv")
    old_participation_path = os.path.join(RESULTS_DIR, "attack_injection_participation.csv")
    with open(old_summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for cell, source in REUSED_SOURCE_LABEL.items():
                if row["config"] == source:
                    summary_rows.append({
                        "config": cell, "round": int(row["round"]),
                        "accuracy": float(row["accuracy"]), "f1": float(row["f1"]),
                        "margin_min": float(row["margin_min"]), "margin_max": float(row["margin_max"]),
                        "margin_mean": float(row["margin_mean"]),
                    })
    with open(old_participation_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for cell, source in REUSED_SOURCE_LABEL.items():
                if row["config"] == source:
                    participation_rows.append({
                        "config": cell, "round": int(row["round"]),
                        "participating_clients": int(row["participating_clients"]),
                        "num_clients": int(row["num_clients"]),
                    })
    # 無防禦兩格沒有排除邏輯，逐輪都是空清單——明確寫出來，方便跟 ERR/LFR 兩格對照
    for cell in REUSED_NO_DEFENSE_MODELS:
        for r in range(1, NUM_ROUNDS + 1):
            exclusion_rows.append({
                "config": cell, "round": r, "excluded_client_ids": "[]",
                "err_flagged_idx": "", "lfr_flagged_idx": "", "union_size": 0, "num_clients": NUM_CLIENTS,
            })

    # --- 新跑兩組 ERR/LFR ---
    for label, malicious_ids, flip_rate in FRESH_ERR_LFR_CONFIGS:
        run_one_err_lfr(label, malicious_ids, flip_rate, summary_rows, participation_rows, exclusion_rows)

    summary_path = os.path.join(RESULTS_DIR, "err_lfr_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "round", "accuracy", "f1", "margin_min", "margin_max", "margin_mean",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[Info] wrote {summary_path} ({len(summary_rows)} rows)")

    participation_path = os.path.join(RESULTS_DIR, "err_lfr_participation.csv")
    with open(participation_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "round", "participating_clients", "num_clients"])
        writer.writeheader()
        writer.writerows(participation_rows)
    print(f"[Info] wrote {participation_path} ({len(participation_rows)} rows)")

    exclusion_path = os.path.join(RESULTS_DIR, "err_lfr_exclusions.csv")
    with open(exclusion_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "round", "excluded_client_ids", "err_flagged_idx", "lfr_flagged_idx",
            "union_size", "num_clients",
        ])
        writer.writeheader()
        writer.writerows(exclusion_rows)
    print(f"[Info] wrote {exclusion_path} ({len(exclusion_rows)} rows)")

    # --- 四格的 round-10 模型路徑，統一跑最終報告（test_natural/test_rare） ---
    cell_model_paths = dict(REUSED_NO_DEFENSE_MODELS)
    for label, _, _ in FRESH_ERR_LFR_CONFIGS:
        cell_model_paths[label] = os.path.join(BASE_DIR, "outputs", "models", f"err_lfr_{label}", f"global_model_round_{NUM_ROUNDS}.ubj")

    final_rows = []
    build_final_report(cell_model_paths, final_rows)
    final_report_path = os.path.join(RESULTS_DIR, "err_lfr_final_report.csv")
    with open(final_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cell", "model_path", "test_natural_accuracy", "test_natural_precision",
            "test_natural_recall", "test_natural_f1",
        ])
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[Info] wrote {final_report_path} ({len(final_rows)} rows)")


if __name__ == "__main__":
    main()
