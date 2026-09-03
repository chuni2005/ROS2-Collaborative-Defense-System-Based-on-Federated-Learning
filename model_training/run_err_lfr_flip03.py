"""跑一組追加設定：攻擊 + ERR/LFR，但把 label_flip_rate 從既有的 1.0 降為
0.3，其餘設定比照既有的四格矩陣實驗，重用 run_err_lfr_experiment.py 的
訓練與報告邏輯。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 要改跑哪個 cell 標籤、哪些節點惡意、翻轉比例，改這三個常數；其餘設定沿用
# run_err_lfr_experiment.py 的共用參數（透過 import 存取，不在本檔案重複定義）。
import csv
import os
import re

import run_err_lfr_experiment as exp

CELL_LABEL = "attack_flip0.3_err_lfr"
MALICIOUS_IDS = {1}
FLIP_RATE = 0.3

# ==================== 區塊 2：log 解析規則 ====================
# 綁死 server.py 印出來的 candidate A 與 client impact 這兩種 log 行格式。
CANDIDATE_A_RE = re.compile(
    r"\[ErrLfr\] Round (\d+) candidate A \(all \d+ clients\) on validation "
    r"\(chunk_6\): accuracy=([\d.]+) logloss=([\d.]+)"
)
CLIENT_IMPACT_RE = re.compile(
    r"\[ErrLfr\]\[ClientImpact\] Round (\d+) client_id=(\S+) \(cid=(\S+), malicious=(\S+)\) "
    r"B_i accuracy=([\d.]+) logloss=([\d.]+) err_impact=(-?[\d.]+) lfr_impact=(-?[\d.]+)"
)


def main():
    """跑這一格新設定的完整 10 輪訓練，把逐輪 summary/participation/exclusion
    以及新增的 candidate_a/client_impact 全部寫進 results/，最後對 round-10
    模型算一次 test_natural/test_rare 的最終報告數字。
    """
    # ==================== 區塊 3：跑訓練與寫出基本三份 csv ====================
    # 重用 run_err_lfr_experiment.py 的 run_one_err_lfr()，跑完整 10 輪訓練並解析
    # 逐輪 summary/participation/exclusion，再各自寫成一份 csv。
    os.makedirs(exp.RESULTS_DIR, exist_ok=True)
    summary_rows, participation_rows, exclusion_rows = [], [], []

    exp.run_one_err_lfr(CELL_LABEL, MALICIOUS_IDS, FLIP_RATE,
                         summary_rows, participation_rows, exclusion_rows)

    def write_csv(rows, filename, fieldnames):
        path = os.path.join(exp.RESULTS_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Info] wrote {path} ({len(rows)} rows)")

    write_csv(summary_rows, "err_lfr_flip03_summary.csv",
              ["config", "round", "accuracy", "f1", "margin_min", "margin_max", "margin_mean"])
    write_csv(participation_rows, "err_lfr_flip03_participation.csv",
              ["config", "round", "participating_clients", "num_clients"])
    write_csv(exclusion_rows, "err_lfr_flip03_exclusions.csv",
              ["config", "round", "excluded_client_ids", "err_flagged_idx", "lfr_flagged_idx",
               "union_size", "num_clients"])

    # ==================== 區塊 4：額外解析（候選模型 A 與節點 impact） ====================
    # --- 額外解析：候選模型 A 逐輪分數、每輪各節點的 err_impact/lfr_impact ---
    log_path = os.path.join(exp.BASE_DIR, "outputs", "logs", f"err_lfr_{CELL_LABEL}", "server.log")
    with open(log_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    candidate_a_rows = []
    for m in CANDIDATE_A_RE.finditer(log_text):
        rnd, acc, logloss = m.groups()
        candidate_a_rows.append({
            "config": CELL_LABEL, "round": int(rnd),
            "accuracy": float(acc), "logloss": float(logloss),
        })
    write_csv(candidate_a_rows, "err_lfr_flip03_candidate_a.csv",
              ["config", "round", "accuracy", "logloss"])

    client_impact_rows = []
    for m in CLIENT_IMPACT_RE.finditer(log_text):
        rnd, client_id, cid, malicious, bi_acc, bi_logloss, err_impact, lfr_impact = m.groups()
        client_impact_rows.append({
            "config": CELL_LABEL, "round": int(rnd), "client_id": client_id, "cid": cid,
            "malicious": malicious, "bi_accuracy": float(bi_acc), "bi_logloss": float(bi_logloss),
            "err_impact": float(err_impact), "lfr_impact": float(lfr_impact),
        })
    write_csv(client_impact_rows, "err_lfr_flip03_client_impact.csv",
              ["config", "round", "client_id", "cid", "malicious", "bi_accuracy", "bi_logloss",
               "err_impact", "lfr_impact"])

    # ==================== 區塊 5：最終報告 ====================
    # --- 最終報告：test_natural 整體指標、test_rare 逐攻擊類型 recall ---
    model_path = os.path.join(exp.BASE_DIR, "outputs", "models", f"err_lfr_{CELL_LABEL}",
                               f"global_model_round_{exp.NUM_ROUNDS}.ubj")
    final_rows = []
    exp.build_final_report({CELL_LABEL: model_path}, final_rows)
    write_csv(final_rows, "err_lfr_flip03_final_report.csv",
              ["cell", "model_path", "test_natural_accuracy", "test_natural_precision",
               "test_natural_recall", "test_natural_f1"])


if __name__ == "__main__":
    main()
