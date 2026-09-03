"""補寫 err_lfr_no_attack_err_lfr／err_lfr_attack_err_lfr（flip_rate=1.0，四格矩陣
主線）的 results/err_lfr_candidate_a.csv、results/err_lfr_client_impact.csv——
run_err_lfr_experiment.py 當初沒有解析 server.log 裡候選模型 A／節點 err_impact、
lfr_impact 那幾行，這裡從 run_err_lfr_flip03.py 抽出同一套解析邏輯，改成讀
outputs/logs/ 底下兩個既有 log 檔案，不重新訓練、不啟動 server.py／client.py。
"""
import csv
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")

CELLS = ["no_attack_err_lfr", "attack_err_lfr"]

CANDIDATE_A_RE = re.compile(
    r"\[ErrLfr\] Round (\d+) candidate A \(all \d+ clients\) on validation "
    r"\(chunk_6\): accuracy=([\d.]+) logloss=([\d.]+)"
)
CLIENT_IMPACT_RE = re.compile(
    r"\[ErrLfr\]\[ClientImpact\] Round (\d+) client_id=(\S+) \(cid=(\S+), malicious=(\S+)\) "
    r"B_i accuracy=([\d.]+) logloss=([\d.]+) err_impact=(-?[\d.]+) lfr_impact=(-?[\d.]+)"
)


def write_csv(rows, filename, fieldnames):
    """把 rows 整份覆寫成 results/filename（"w" 模式，不是 append）。"""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Info] wrote {path} ({len(rows)} rows)")


def main():
    """對 CELLS 裡每一組的 server.log 各跑一次正規表示式擷取，兩組結果合併後
    各自寫成一份 csv。log 檔不存在時印提示訊息跳過該組，不中斷整支腳本。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    candidate_a_rows = []
    client_impact_rows = []

    for cell in CELLS:
        log_path = os.path.join(BASE_DIR, "outputs", "logs", f"err_lfr_{cell}", "server.log")
        if not os.path.exists(log_path):
            print(f"[缺檔] {log_path} 不存在，略過 {cell}。")
            continue
        with open(log_path, "r", encoding="utf-8") as f:
            log_text = f.read()

        for m in CANDIDATE_A_RE.finditer(log_text):
            rnd, acc, logloss = m.groups()
            candidate_a_rows.append({
                "config": cell, "round": int(rnd),
                "accuracy": float(acc), "logloss": float(logloss),
            })

        for m in CLIENT_IMPACT_RE.finditer(log_text):
            rnd, client_id, cid, malicious, bi_acc, bi_logloss, err_impact, lfr_impact = m.groups()
            client_impact_rows.append({
                "config": cell, "round": int(rnd), "client_id": client_id, "cid": cid,
                "malicious": malicious, "bi_accuracy": float(bi_acc), "bi_logloss": float(bi_logloss),
                "err_impact": float(err_impact), "lfr_impact": float(lfr_impact),
            })

    write_csv(candidate_a_rows, "err_lfr_candidate_a.csv",
              ["config", "round", "accuracy", "logloss"])
    write_csv(client_impact_rows, "err_lfr_client_impact.csv",
              ["config", "round", "client_id", "cid", "malicious", "bi_accuracy", "bi_logloss",
               "err_impact", "lfr_impact"])


if __name__ == "__main__":
    main()
