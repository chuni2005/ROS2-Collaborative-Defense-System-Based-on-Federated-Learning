"""跑留一法（leave-one-out）量測：訓練「全部節點都在」（候選 A）與 5 次
「拿掉一個節點」（候選 B_1~B_5）的模型，比較 F1 差異，看拿掉哪個節點影響
最大。這只是量測，不是真正的 ERR/LFR 剔除機制。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 要改留一法要排除哪些節點，改 ALL_CLIENTS（CONFIGS 由它自動生成）；
# LEAF_SCALE 是掃描過的固定值，其餘是共用路徑與訓練參數。
import csv
import os
import re
import shutil
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
NUM_ROUNDS = 10
SERVER_ADDRESS = "127.0.0.1:8080"
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_data", "chunk_6.csv")
# 是 0.5，不是 1/client 數（0.2）——掃描過 {1/5, 1/3, 1/2, 1} 四個值：前
# 三個都穩定收斂，而且 F1／reconnaissance recall 都隨係數增加單調變好，
# 一路到 1/2（三個穩定值裡最高的）都還在變好，所以 1/2 是實際測過的四個
# 值裡最好的，不是「除以 client 數」這個理論推出來的值。leaf_scale=1（不
# 縮放）會重現原本的 margin 暴衝問題。完整掃描結果見 notes/13a-leaf-scale-fix.md。
LEAF_SCALE = 0.5

ALL_CLIENTS = [1, 2, 3, 4, 5]
CONFIGS = [("A", ALL_CLIENTS)] + [
    (f"B_{i}", [c for c in ALL_CLIENTS if c != i]) for i in ALL_CLIENTS
]


# ==================== 區塊 2：log 解析規則 ====================
# 這個正則表達式綁死 server.py 的 log 字串格式（XGBoostBaggingStrategy.
# aggregate_fit() 裡那行 print）——server.py 的 log 文字如果之後改了措辭，
# 這裡會安靜地配不到任何東西（finditer 回傳空結果，不會報錯），不會有任何
# 提示告訴你資料是空的，這是已知的脆弱點（notes/16-code-review-guide.md
# Part B 第 3 項）。
ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) bagging-merged \d+ client models "
    r"\(accuracy=([\d.]+), f1=([\d.]+), margin=\[(-?[\d.]+), (-?[\d.]+)\], "
    r"margin_mean=(-?[\d.]+)\)"
)


# ==================== 區塊 3：子行程收尾工具 ====================
# 每組留一法設定跑完（或逾時）之後，用來把 server/client 子行程都關掉的共用函式。
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


def run_one(label, client_ids):
    """跑一組留一法設定（候選 A 或某個排除節點 i 的 B_i）的完整 10 輪
    聯邦 bagging 訓練，解析伺服器 log 取得逐輪指標。

    輸入：label 是這組設定的檔名/資料夾用標籤（"A" 或 "B_i"）；client_ids
    是這組設定實際要啟動的節點編號列表。
    輸出：這組設定逐輪的指標，一個 list of dict（每輪一筆，含 config、
    excluded_client、round、accuracy、f1、margin_min、margin_max、
    margin_mean）。副作用包括寫 log 到 outputs/logs/loo_<label>/、把訓練出的模型
    存到 outputs/models/loo_<label>/。各階段見下方區塊 4-7 註解。
    """
    num_clients = len(client_ids)
    print(f"\n===== {label}: clients={client_ids} (num_clients={num_clients}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, "outputs", "models", f"loo_{label}")
    log_dir = os.path.join(BASE_DIR, "outputs", "logs", f"loo_{label}")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # ==================== 區塊 4：啟動 server ====================
    # --- 啟動 server（num_clients 對齊實際節點數） ---
    server_log_path = os.path.join(log_dir, "server.log")
    server_log = open(server_log_path, "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [
            sys.executable, os.path.join(BASE_DIR, "server.py"),
            f"--model_dir={model_dir}",
            f"--num_clients={num_clients}",
            f"--num_rounds={NUM_ROUNDS}",
            f"--server_address={SERVER_ADDRESS}",
            f"--validation_data_path={VAL_DATA_PATH}",
            "--aggregation=bagging",
            f"--leaf_scale={LEAF_SCALE}",
        ],
        stdout=server_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
    )
    time.sleep(3)  # 讓 server 先把 gRPC port 綁好，client 才不會連線失敗

    # ==================== 區塊 5：啟動 client ====================
    # --- 啟動 client_ids 裡指定的每個 client ---
    client_procs = []
    client_logs = []
    for i in client_ids:
        data_path = os.path.join(BASE_DIR, "split_data", f"chunk_{i}.csv")
        client_log = open(os.path.join(log_dir, f"client_{i}.log"), "w", encoding="utf-8")
        client_logs.append(client_log)
        proc = subprocess.Popen(
            [
                sys.executable, os.path.join(BASE_DIR, "client.py"),
                f"--client_id={i}",
                f"--data_path={data_path}",
                f"--server_address={SERVER_ADDRESS}",
                "--aggregation=bagging",
            ],
            stdout=client_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
        )
        client_procs.append(proc)

    # ==================== 區塊 6：等待與收拾 ====================
    # --- 等 server 結束或逾時、收拾所有子行程 ---
    start = time.time()
    timeout_s = 300  # 觀察到單組實際約 40-85 秒跑完，這是留給異常情況的安全值
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

    # ==================== 區塊 7：解析 log 並回傳 ====================
    # --- 解析 server log 取得逐輪指標並回傳 ---
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    rows = []
    for m in ROUND_RE.finditer(server_text):
        rnd, acc, f1, mmin, mmax, mmean = m.groups()
        rows.append({
            "config": label, "excluded_client": (None if label == "A" else int(label.split("_")[1])),
            "round": int(rnd), "accuracy": float(acc), "f1": float(f1),
            "margin_min": float(mmin), "margin_max": float(mmax), "margin_mean": float(mmean),
        })
    return rows


# ==================== 區塊 8：主流程／寫出結果與比較 ====================
# 依序對 CONFIGS 每組呼叫 run_one()，寫出彙整 csv，再算並印出每個 B_i 相對 A 的 impact。
def main():
    """依序對 CONFIGS（候選 A 加上每個排除單一節點的 B_i）呼叫
    run_one()，把全部結果寫成 results/loo_impact_summary.csv，再取每組
    第 10 輪的 F1，印出每個 B_i 相對 A 的 impact
    （Impact_i = F1_A − F1_Bi）。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for label, client_ids in CONFIGS:
        all_rows.extend(run_one(label, client_ids))

    summary_path = os.path.join(RESULTS_DIR, "loo_impact_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "excluded_client", "round", "accuracy", "f1",
            "margin_min", "margin_max", "margin_mean",
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[Info] wrote {summary_path} ({len(all_rows)} rows)")

    round10 = {r["config"]: r for r in all_rows if r["round"] == NUM_ROUNDS}
    f1_a = round10["A"]["f1"]
    print(f"\nCandidate A round-10 F1 = {f1_a:.6f}")
    print(f"{'config':10s} {'F1':>10s} {'Impact(A-Bi)':>14s}")
    for label, _ in CONFIGS[1:]:
        f1_b = round10[label]["f1"]
        print(f"{label:10s} {f1_b:10.6f} {f1_a - f1_b:14.6f}")


if __name__ == "__main__":
    main()
