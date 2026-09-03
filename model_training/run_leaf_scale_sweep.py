"""掃描 leaf_scale ∈ {1/5, 1/3, 1/2, 1} 四個候選值，各跑一次完整的 10 輪
bagging 訓練，找出哪個值能讓合併穩定收斂、不會讓預測值爆掉。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 要改掃描哪些 leaf_scale 候選值，改 SWEEP_VALUES；其餘是共用路徑與訓練固定參數。
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

# ==================== 區塊 2：log 解析規則 ====================
# 綁死 server.py 在 bagging 模式下印出來的 log 文字格式，用來抓每輪的 accuracy/f1/margin。
ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) bagging-merged \d+ client models "
    r"\(accuracy=([\d.]+), f1=([\d.]+), margin=\[(-?[\d.]+), (-?[\d.]+)\], "
    r"margin_mean=(-?[\d.]+)\)"
)


# ==================== 區塊 3：子行程收尾工具 ====================
# 每組候選值跑完（或逾時）之後，用來把 server/client 子行程都關掉的共用函式。
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


def run_one(label, w, summary_rows):
    """跑一組 leaf_scale 設定的完整 10 輪聯邦 bagging 訓練，解析伺服器
    log 取得逐輪指標，並對第 10 輪模型跑一次逐攻擊類型 recall 分析。

    輸入：label 是這組設定的檔名/資料夾用標籤（例如 "1-2"）；w 是實際
    傳給 server.py --leaf_scale 的縮放係數；summary_rows 是呼叫端傳進來
    的 list，這個函式會直接把這組設定的逐輪結果 append 進去（就地修改，
    沒有回傳值）。
    輸出：無回傳值。副作用包括：寫 log 到 outputs/logs/leaf_scale_<label>/、把
    訓練出的模型存到 outputs/models/leaf_scale_<label>/、把逐攻擊類型 recall 寫到
    results/leaf_scale_<label>_recall.txt。各階段見下方區塊 4-8 註解。
    """
    print(f"\n===== leaf_scale={w:.6f} (label={label}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, "outputs", "models", f"leaf_scale_{label}")
    log_dir = os.path.join(BASE_DIR, "outputs", "logs", f"leaf_scale_{label}")
    os.makedirs(log_dir, exist_ok=True)
    # 每個掃描值都用全新的 model_dir——不然 initialize_parameters() 會接著
    # 前一個掃描值留下的 global_model_latest.ubj 繼續訓練，不是從頭開始
    if os.path.isdir(model_dir):
        import shutil
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # ==================== 區塊 4：啟動 server ====================
    # --- 啟動 server（帶這組的 --leaf_scale） ---
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

    # ==================== 區塊 5：啟動 client ====================
    # --- 啟動全部 5 個 client ---
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

    # ==================== 區塊 6：等待與收拾 ====================
    # --- 等 server 結束或逾時、收拾所有子行程 ---
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

    # ==================== 區塊 7：解析 log ====================
    # --- 解析 server log 取得逐輪 accuracy/f1/margin ---
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    for m in ROUND_RE.finditer(server_text):
        rnd, acc, f1, mmin, mmax, mmean = m.groups()
        summary_rows.append({
            "leaf_scale_label": label, "leaf_scale": w, "round": int(rnd),
            "accuracy": float(acc), "f1": float(f1),
            "margin_min": float(mmin), "margin_max": float(mmax), "margin_mean": float(mmean),
        })

    # ==================== 區塊 8：後續分析 ====================
    # --- 對第 10 輪的模型跑逐攻擊類型 recall 分析 ---
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


# ==================== 區塊 9：主流程／寫出結果 ====================
# 依序對 SWEEP_VALUES 每組呼叫 run_one()，全部跑完後把彙整出的結果寫成一份 csv。
def main():
    """依序對 SWEEP_VALUES 裡每組 (label, w) 呼叫 run_one()，全部跑完後
    把彙整出的 summary_rows 寫成 results/leaf_scale_summary.csv。
    """
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
