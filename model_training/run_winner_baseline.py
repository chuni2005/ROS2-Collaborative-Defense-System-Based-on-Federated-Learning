"""跑一次完整的 10 輪贏者全拿 baseline，重新產生 baseline 數字。

原理：這不是獨立的聚合機制，是把既有的 server.py（--aggregation=winner）
跟 client.py 組合起來，跑一次完整流程再把結果整理成檔案——跟
run_leaf_scale_sweep.py 是同一套執行框架，差別只在這裡只跑一組設定
（贏者全拿），不是掃描多個係數。

輸入：磁碟上既有的 split_data/chunk_1..6.csv。
輸出：伺服器 log 解析出來的逐輪 accuracy/F1（印在終端機），以及第 10 輪
模型的逐攻擊類型 recall（results/baseline_reseed_recall.txt）。

怎麼做：啟動 server.py（--aggregation=winner）跟 5 個 client.py 子行程，
跑滿 10 輪；伺服器行程結束後，從它的 log 解析出每一輪贏家的
accuracy/F1 印出來；再對第 10 輪存下來的模型呼叫
analyze_recall_by_attack.py 算逐攻擊類型 recall。

為什麼需要它：split_data 用固定種子重新產生過一次（SPLIT_SEED=42）之後，
舊切分上的 baseline 數字沒辦法用同一份資料重現，這支腳本在新切分上補跑
一次，讓原本的 baseline 頭條數字有對應的新資料版本可以對照。

完整調查過程見 notes/12-baseline.md、notes/12b-branch-delta.md。
"""
import os
import re
import shutil
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
NUM_CLIENTS = 5
NUM_ROUNDS = 10
SERVER_ADDRESS = "127.0.0.1:8080"
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_data", "chunk_6.csv")

ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) kept the highest-F1 model \(client \S+, "
    r"client_id=(\S+), accuracy=([\d.]+), f1=([\d.]+)\)"
)


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


def main():
    """跑一次完整的 10 輪贏者全拿（--aggregation=winner）聯邦訓練，解析
    伺服器 log 取得每輪贏家的 accuracy/f1，並對第 10 輪模型跑一次逐攻擊
    類型 recall 分析。

    輸入：無參數，設定寫死在檔案開頭常數。
    輸出：無回傳值。副作用包括：寫 log 到 logs/baseline_reseed/、把訓練
    出的模型存到 model_baseline_reseed/、把逐攻擊類型 recall 寫到
    results/baseline_reseed_recall.txt，並把逐輪結果印到終端機。

    怎麼做分成幾個階段：
        # --- 啟動 server（winner 模式） ---
        # --- 啟動全部 5 個 client（winner 模式） ---
        # --- 等 server 結束或逾時、收拾所有子行程 ---
        # --- 解析 server log 取得每輪贏家的 accuracy/f1 ---
        # --- 對第 10 輪的模型跑逐攻擊類型 recall 分析 ---
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_dir = os.path.join(BASE_DIR, "model_baseline_reseed")
    log_dir = os.path.join(BASE_DIR, "logs", "baseline_reseed")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # --- 啟動 server（winner 模式） ---
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
            "--aggregation=winner",
        ],
        stdout=server_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
    )
    time.sleep(3)

    # --- 啟動全部 5 個 client（winner 模式） ---
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
                "--aggregation=winner",
            ],
            stdout=client_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
        )
        client_procs.append(proc)

    # --- 等 server 結束或逾時、收拾所有子行程 ---
    start = time.time()
    timeout_s = 300
    while server_proc.poll() is None:
        if time.time() - start > timeout_s:
            print(f"[Warning] server did not finish within {timeout_s}s, killing.")
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

    # --- 解析 server log 取得每輪贏家的 accuracy/f1 ---
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    for m in ROUND_RE.finditer(server_text):
        rnd, cid, acc, f1 = m.groups()
        print(f"round {rnd}: client_id={cid} accuracy={acc} f1={f1}")

    # --- 對第 10 輪的模型跑逐攻擊類型 recall 分析 ---
    round10_model = os.path.join(model_dir, f"global_model_round_{NUM_ROUNDS}.ubj")
    if os.path.exists(round10_model):
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "analyze_recall_by_attack.py"),
             f"--model_path={round10_model}", f"--val_data_path={VAL_DATA_PATH}"],
            cwd=BASE_DIR, capture_output=True, text=True,
        )
        recall_path = os.path.join(RESULTS_DIR, "baseline_reseed_recall.txt")
        with open(recall_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
            if result.returncode != 0:
                f.write("\n[stderr]\n" + result.stderr)
        print(f"[Info] wrote {recall_path}")
        print(result.stdout)


if __name__ == "__main__":
    main()
