"""跑一次完整的 10 輪贏者全拿（winner-take-all）baseline，重新產生 baseline 數字。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 這支腳本只跑單一組設定，會影響輸出的參數都是這裡的模組層級常數。
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

# ==================== 區塊 2：log 解析規則 ====================
# 綁死 server.py 在贏者全拿模式下印出來的 log 文字格式，用來抓每輪贏家的 accuracy/f1。
ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) kept the highest-F1 model \(client \S+, "
    r"client_id=(\S+), accuracy=([\d.]+), f1=([\d.]+)\)"
)


# ==================== 區塊 3：子行程收尾工具 ====================
# 訓練跑完（或逾時）之後，用來把 server/client 子行程都關掉的共用函式。
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
    輸出：無回傳值。副作用包括：寫 log 到 outputs/logs/baseline_reseed/、把訓練
    出的模型存到 outputs/models/baseline_reseed/、把逐攻擊類型 recall 寫到
    results/baseline_reseed_recall.txt，並把逐輪結果印到終端機。各階段見下方
    區塊 4-8 註解。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_dir = os.path.join(BASE_DIR, "outputs", "models", "baseline_reseed")
    log_dir = os.path.join(BASE_DIR, "outputs", "logs", "baseline_reseed")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # ==================== 區塊 4：啟動 server ====================
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

    # ==================== 區塊 5：啟動 client ====================
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

    # ==================== 區塊 6：等待與收拾 ====================
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

    # ==================== 區塊 7：解析 log ====================
    # --- 解析 server log 取得每輪贏家的 accuracy/f1 ---
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    for m in ROUND_RE.finditer(server_text):
        rnd, cid, acc, f1 = m.groups()
        print(f"round {rnd}: client_id={cid} accuracy={acc} f1={f1}")

    # ==================== 區塊 8：後續分析／寫出結果 ====================
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
