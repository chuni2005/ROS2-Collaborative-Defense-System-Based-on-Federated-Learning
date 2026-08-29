"""掃描 leaf_scale 這個縮放係數，測哪個值能讓 bagging 合併穩定收斂。

原理：bagging 合併時，5 個節點每輪各自獨立對同一個全域模型修正，伺服器
直接把 5 份修正加總（不是取平均），等於把學習率放大了將近 5 倍，逐輪
疊加下去會讓預測值爆掉。leaf_scale 是拿來抵銷這個放大效果的縮放係數，
掃描不同值就是為了找出哪個值最好（機制細節見 server.py 的
scale_leaf_values()）。

輸入：磁碟上既有的 split_data/chunk_1..6.csv（不重新切分，理由見下）；
掃描 leaf_scale ∈ {1/5, 1/3, 1/2, 1} 四個候選值。
輸出：results/leaf_scale_summary.csv（四組 × 10 輪的逐輪
accuracy/F1/margin），以及每組第 10 輪模型的逐攻擊類型 recall
（results/leaf_scale_<label>_recall.txt）。

怎麼做：對每個候選值，各自跑一次完整的 10 輪聯邦 bagging 訓練，各自用一個
全新的 model_dir（不然 initialize_parameters() 會接著前一個候選值留下的
模型繼續訓練）跟自己的 log 目錄（logs/leaf_scale_<label>/）。逐輪的
accuracy/F1/margin 是從伺服器自己的 stdout 解析出來的（「[Info] Round N
bagging-merged ...」這幾行 log），四組跑完後彙整成一份 csv；另外對每組
第 10 輪的模型呼叫 analyze_recall_by_attack.py 算逐攻擊類型 recall。

為什麼刻意不重新切分：split_data/chunk_*.csv 已經是前一次執行留下來的，
而且 split.py 的切分沒有固定隨機種子——如果在掃描不同 leaf_scale 值之間
重新切分，會把「leaf_scale 造成的效果」跟「換了一份資料切分造成的效果」
混在一起，分不清楚是哪個原因，所以四個候選值全部沿用磁碟上既有的同一份
資料訓練與評估。

完整調查過程見 notes/13a-bagging-baseline.md、notes/12-baseline.md。
"""
import csv
import os
import re
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
NUM_CLIENTS = 5
NUM_ROUNDS = 10
SERVER_ADDRESS = "127.0.0.1:8080"
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_data", "chunk_6.csv")

# （標籤用在資料夾／檔名裡，實際傳進 --leaf_scale 的是 w 這個數值）
SWEEP_VALUES = [
    ("1-5", 1.0 / 5.0),
    ("1-3", 1.0 / 3.0),
    ("1-2", 1.0 / 2.0),
    ("1-1", 1.0),
]

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


def run_one(label, w, summary_rows):
    print(f"\n===== leaf_scale={w:.6f} (label={label}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, f"model_leaf_scale_{label}")
    log_dir = os.path.join(BASE_DIR, "logs", f"leaf_scale_{label}")
    os.makedirs(log_dir, exist_ok=True)
    # 每個掃描值都用全新的 model_dir——不然 initialize_parameters() 會接著
    # 前一個掃描值留下的 global_model_latest.ubj 繼續訓練，不是從頭開始
    if os.path.isdir(model_dir):
        import shutil
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
            "--aggregation=bagging",
            f"--leaf_scale={w}",
        ],
        stdout=server_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
    )
    time.sleep(3)

    client_procs = []
    client_logs = []
    for i in range(1, NUM_CLIENTS + 1):
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
    timeout_s = 300
    while server_proc.poll() is None:
        if time.time() - start > timeout_s:
            print(f"[Warning] leaf_scale={w} server did not finish within {timeout_s}s, killing.")
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
    for m in ROUND_RE.finditer(server_text):
        rnd, acc, f1, mmin, mmax, mmean = m.groups()
        summary_rows.append({
            "leaf_scale_label": label, "leaf_scale": w, "round": int(rnd),
            "accuracy": float(acc), "f1": float(f1),
            "margin_min": float(mmin), "margin_max": float(mmax), "margin_mean": float(mmean),
        })

    round10_model = os.path.join(model_dir, f"global_model_round_{NUM_ROUNDS}.ubj")
    recall_path = os.path.join(RESULTS_DIR, f"leaf_scale_{label}_recall.txt")
    if os.path.exists(round10_model):
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "analyze_recall_by_attack.py"),
             f"--model_path={round10_model}", f"--val_data_path={VAL_DATA_PATH}"],
            cwd=BASE_DIR, capture_output=True, text=True,
        )
        with open(recall_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            if result.returncode != 0:
                f.write("\n[stderr]\n" + result.stderr)
        print(f"[Info] wrote {recall_path}")
    else:
        print(f"[Warning] round-10 model not found at {round10_model}, skipping recall analysis.")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    for label, w in SWEEP_VALUES:
        run_one(label, w, summary_rows)

    summary_path = os.path.join(RESULTS_DIR, "leaf_scale_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "leaf_scale_label", "leaf_scale", "round", "accuracy", "f1",
            "margin_min", "margin_max", "margin_mean",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[Info] wrote {summary_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
