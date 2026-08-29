"""Leave-one-out impact of each honest client on the FIXED bagging pipeline
(leaf_scale=0.5, see notes/13a-leaf-scale-fix.md -- w=0.5 finalized there).

中文導讀：
吃什麼：這支腳本不接受參數，設定寫死在檔案開頭常數（`LEAF_SCALE`、
    `ALL_CLIENTS` 等）——實際的輸入是磁碟上既有的 `split_data/chunk_1.csv`
    ~`chunk_6.csv`（5 個 client 的訓練資料 + 1 份伺服器驗證資料），不會重新
    切分。
吐什麼：`results/loo_impact_summary.csv`（六組設定 × 10 輪的逐輪
    accuracy/f1/margin），以及印在終端機上的一張 impact 表（`config`／`F1`／
    `Impact(A-Bi)`）。
中間轉換：對六組設定（候選 A：全部 5 個 client；候選 B_1~B_5：各排除一個
    client）分別跑一次完整的 10 輪聯邦 bagging 訓練（真的啟動 server.py +
    對應數量的 client.py 子行程，不是模擬），全部跑完之後比較每組的
    round-10 F1。
在流程裡的位置：這是 ERR/LFR 的核心動作（拿掉一個節點、重新聚合、比較跟
    「全部保留」的差異）已經在跑，但目前只是**量測**——把量出來的差異拿去
    跟「沒有拿掉任何節點」的雜訊底噪比較，不是拿去**決定要不要真的剔除某個
    節點**。要變成真正的 ERR/LFR，還缺：(1) 每一輪即時算 impact，不是跑完
    整個 10 輪才算一次；(2) 依 impact 排序，選出最大的 c 個節點；(3) 把這
    c 個節點從這一輪的聚合裡真的排除掉，不是像現在這樣拿掉之後重跑一次
    獨立的訓練。

Redo of the LOO measurement in notes/12-baseline.md's "留一法（bagging LOO）"
section, which used Flower's OFFICIAL aggregate() applied ONCE to each
client's already-cumulative round-10 model -- that reproduces the exact bug
notes/13a-bagging-baseline.md found (aggregate() reads num_parallel_tree=1,
so it silently takes only each model's stale first tree, not a real 10-round
federated run). The impact numbers from that run (all 5 negative, full range
0.000016) are noise from a broken merge, not a measurement of real per-client
signal -- this script re-measures it on the actual fixed pipeline.

Candidate A: 10 real federated bagging rounds with all 5 honest clients.
Candidate B_i: same, but client i excluded from training entirely (4
clients, --num_clients=4 so Flower's min_fit_clients matches who's actually
connected). Both use the same split_data/chunk_*.csv already on disk (no
re-split -- see run_leaf_scale_sweep.py's docstring for why) and the same
leaf_scale=0.5, so the only difference between A and each B_i is the
presence/absence of that one client's data and trees -- isolates the
leave-one-out effect from any confound in the scaling policy itself.

Impact_i = F1_A - F1_Bi, same sign convention as notes/12-baseline.md.
"""
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
# 0.5, not 1/num_clients (0.2) -- notes/13a-leaf-scale-fix.md swept
# {1/5, 1/3, 1/2, 1}: the first three all converge stably, and F1/
# reconnaissance-recall both still improve monotonically up to 1/2 (the
# highest of the three stable values tested), so 1/2 is the best of the
# four actually tried, not a value derived from a "divide by client count"
# theory. leaf_scale=1 (unscaled) reproduces the original margin-explosion
# bug -- see server.py's scale_leaf_values() docstring.
LEAF_SCALE = 0.5

ALL_CLIENTS = [1, 2, 3, 4, 5]
CONFIGS = [("A", ALL_CLIENTS)] + [
    (f"B_{i}", [c for c in ALL_CLIENTS if c != i]) for i in ALL_CLIENTS
]


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


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_one(label, client_ids):
    num_clients = len(client_ids)
    print(f"\n===== {label}: clients={client_ids} (num_clients={num_clients}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, f"model_loo_{label}")
    log_dir = os.path.join(BASE_DIR, "logs", f"loo_{label}")
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


def main():
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
