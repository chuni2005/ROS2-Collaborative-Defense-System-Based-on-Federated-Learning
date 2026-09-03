"""載入既有集中式模型（outputs/models/centralized_reseed.ubj，不重新訓練）在
test_natural.csv／test_rare.csv 上重新評估，補上跟四格矩陣其餘三組直接可比的數字。

原本的 results/centralized_reseed_recall.txt 是在 chunk_6.csv 上評估的，跟四格
矩陣用的 test_natural/test_rare 不是同一份資料（攻擊類型樣本數對不上），不能放進
同一張表比較——這裡改用 run_err_lfr_experiment.py 對其他三組用的同一套評估邏輯
重跑一次，寫出 results/centralized_test_rare_recall.txt，並把四組數字並排印出來，
方便判讀集中式的數字有沒有大幅偏離 chunk_6 上量到的水準。
"""
import csv
import os

import xgboost as xgb

import run_err_lfr_experiment as exp
import server as srv

MODEL_PATH = os.path.join(exp.BASE_DIR, "outputs", "models", "centralized_reseed.ubj")

# 四格矩陣最終報告只挑了這三格當「無防禦／有防禦」對照，見
# run_err_lfr_experiment.py 的 err_lfr_final_report.csv。
COMPARISON_CELLS = ["no_attack_no_defense", "attack_no_defense", "attack_err_lfr"]
CELL_LABELS = {
    "no_attack_no_defense": "無攻擊基準",
    "attack_no_defense": "有攻擊無防禦",
    "attack_err_lfr": "有攻擊有防禦",
}

# chunk_6 上量到的水準（centralized_reseed_recall.txt），只當作人工比對的參考基準，
# 不是自動判斷通過/失敗的門檻。
CHUNK6_RECONNAISSANCE_RECALL = 0.9037


def load_existing_cell_numbers():
    """讀 err_lfr_final_report.csv（整體指標）與各 err_lfr_final_recall_<cell>.txt
    （逐攻擊類型 recall），取出 COMPARISON_CELLS 這三格的既有數字，回傳
    {cell: {"f1": float, "recall_rows": [(attack_type, n, value, metric), ...]}}。
    任何一份檔案不存在就整組跳過，並印出提示。
    """
    result = {}
    report_path = os.path.join(exp.RESULTS_DIR, "err_lfr_final_report.csv")
    f1_by_cell = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                f1_by_cell[row["cell"]] = float(row["test_natural_f1"])
    else:
        print(f"[缺檔] {report_path} 不存在。")

    for cell in COMPARISON_CELLS:
        recall_path = os.path.join(exp.RESULTS_DIR, f"err_lfr_final_recall_{cell}.txt")
        if cell not in f1_by_cell or not os.path.exists(recall_path):
            print(f"[缺資料] {cell} 缺 err_lfr_final_report.csv 的列或 {recall_path}，跳過。")
            continue
        rows = []
        with open(recall_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split()
                if len(parts) < 4 or not parts[-2].replace(".", "", 1).isdigit():
                    continue
                *name_parts, n, value, metric = parts
                rows.append((" ".join(name_parts), int(n), float(value), metric))
        result[cell] = {"f1": f1_by_cell[cell], "recall_rows": rows}
    return result


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[缺檔] {MODEL_PATH} 不存在，無法評估。")
        return

    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    # --- test_natural：整體指標，跟其他三組用同一支方法算 ---
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    strategy = srv.XGBoostStrategy(
        model_dir=os.path.join(exp.BASE_DIR, "tmp", "centralized_eval_scratch"),
        num_clients=exp.NUM_CLIENTS,
        val_data_path=exp.VAL_DATA_PATH,
        test_natural_data_path=exp.TEST_NATURAL_PATH,
    )
    overall = strategy._evaluate_model_on_server(model_bytes, dataset="test_natural")

    # --- test_rare：逐攻擊類型 recall/FPR，跟其他三組用同一支方法算 ---
    import pandas as pd
    df_test_rare_raw = pd.read_csv(exp.TEST_RARE_PATH)
    centralized_rows = exp.per_type_recall(bst, df_test_rare_raw)

    # --- 寫檔（格式比照 err_lfr_final_recall_<cell>.txt） ---
    out_path = os.path.join(exp.RESULTS_DIR, "centralized_test_rare_recall.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"model={MODEL_PATH}\n")
        f.write(
            "# 這份檔案的逐攻擊類型 recall/FPR 算在 test_rare.csv 上；\n"
            "# 整體 accuracy/precision/recall/f1 是算在 test_natural.csv 上，\n"
            "# 兩者不是同一份資料，不能直接放在同一列比。\n"
            f"test_natural overall: accuracy={overall['accuracy']:.4f} "
            f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
            f"f1={overall['f1']:.4f}\n\n"
        )
        f.write(f"{'attack_type':30s} {'n':>8s} {'value':>10s}  metric\n")
        for atype, n, rate, metric in centralized_rows:
            f.write(f"{atype:30s} {n:8d} {rate:10.4f}  {metric}\n")
    print(f"[Info] wrote {out_path}")

    # --- 四組並排比較，印在終端機供人工判讀 ---
    existing = load_existing_cell_numbers()
    print()
    print("=== test_natural 整體 F1 ===")
    for cell in COMPARISON_CELLS:
        if cell in existing:
            print(f"  {CELL_LABELS[cell]:10s} f1={existing[cell]['f1']:.4f}")
    print(f"  {'集中式對照':10s} f1={overall['f1']:.4f}")

    print()
    print("=== test_rare 逐攻擊類型 recall/FPR（四組並排） ===")
    all_types = [row[0] for row in centralized_rows]
    header = f"{'attack_type':30s}" + "".join(f"{CELL_LABELS[c]:>14s}" for c in COMPARISON_CELLS) + f"{'集中式對照':>14s}"
    print(header)
    centralized_by_type = {row[0]: row[2] for row in centralized_rows}
    for atype in all_types:
        line = f"{atype:30s}"
        for cell in COMPARISON_CELLS:
            rows = existing.get(cell, {}).get("recall_rows", [])
            match = next((r[2] for r in rows if r[0] == atype), None)
            line += f"{match:14.4f}" if match is not None else f"{'(缺)':>14s}"
        line += f"{centralized_by_type[atype]:14.4f}"
        print(line)

    print()
    recon = centralized_by_type.get("ros2 reconnaissance")
    if recon is None:
        print("[Warning] test_rare.csv 裡沒有 ros2 reconnaissance 這個攻擊類型，無法比對。")
    else:
        diff = recon - CHUNK6_RECONNAISSANCE_RECALL
        print(
            f"[檢查] 集中式模型在 test_rare.csv 上的 ros2 reconnaissance recall = {recon:.4f}，"
            f"與 chunk_6.csv 上量到的 {CHUNK6_RECONNAISSANCE_RECALL:.4f} 相差 {diff:+.4f}。"
        )
        if abs(diff) > 0.1:
            print("[Warning] 差距超過 0.1，跟原本 chunk_6 上的水準不一致，"
                  "先不要把這組數字接進圖4，回報給人工確認再決定。")
        else:
            print("[Info] 差距在 0.1 以內，數量級上跟 chunk_6 的水準一致。")


if __name__ == "__main__":
    main()
