"""跑標籤翻轉攻擊驗收實驗：無攻擊、有攻擊（client 1 標籤翻轉）兩組，都用
bagging、無防禦設定各跑 10 輪，比較兩組表現以確認攻擊有沒有造成可測量的傷害。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 要改跑哪幾組實驗，改 CONFIGS 這個 list；其餘是共用路徑與聯邦訓練的固定參數。
import csv
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
SERVER_ADDRESS = "127.0.0.1:8081"  # 跟 leaf_scale 掃描用的 8080 分開，避免撞埠
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_data", "chunk_6.csv")
LEAF_SCALE = 0.5  # notes/13a-leaf-scale-fix.md 定案的最佳值，兩組都用同一個

# (label, malicious_client_ids, label_flip_rate)
CONFIGS = [
    ("no_attack", set(), 1.0),
    ("attack_c1_full", {1}, 1.0),
]

# ==================== 區塊 2：log 解析規則 ====================
# 這些正則表達式綁死 server.py 印出來的 log 文字格式，用來從 server 的 stdout
# 逐輪抓出這支腳本要記錄的數字。
ROUND_RE = re.compile(
    r"\[Info\] Round (\d+) bagging-merged \d+ client models "
    r"\(accuracy=([\d.]+), f1=([\d.]+), margin=\[(-?[\d.]+), (-?[\d.]+)\], "
    r"margin_mean=(-?[\d.]+)\)"
)
PARTICIPATION_RE = re.compile(
    r"\[Bagging\] Round (\d+): (\d+)/(\d+) client\(s\) participated"
)
CLIENT_SCORE_RE = re.compile(
    r"\[Bagging\]\[ClientScore\] Round (\d+) client_id=(\S+) "
    r"\(cid=(\S+), malicious=(\S+)\) on validation \(chunk_6\): "
    r"f1=([\d.]+) accuracy=([\d.]+) precision=([\d.]+) recall=([\d.]+)"
)


# ==================== 區塊 3：子行程收尾工具 ====================
# 每組實驗結束時，用來把這一輪啟動的 server/client 子行程都關掉的共用函式。
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


def run_one(label, malicious_ids, flip_rate, summary_rows, participation_rows, score_rows):
    """跑一組設定的完整 10 輪聯邦 bagging 訓練，解析伺服器 log 取得逐輪
    指標、每輪參與節點數、每個節點的驗證集分數，並對第 10 輪模型跑一次
    逐攻擊類型 recall 分析。

    輸入：label 是這組設定的檔名/資料夾用標籤；malicious_ids 是這組要標記
    成攻擊節點的 client_id 集合（例如 {1}）；flip_rate 是這些節點的標籤
    翻轉比例；summary_rows/participation_rows/score_rows 是呼叫端傳入的
    list，這個函式會直接把這組的結果 append 進去（就地修改，無回傳值）。
    """
    # ==================== 區塊 4：啟動 server ====================
    # 建立這組設定專用的 model/log 資料夾，再把 server.py 起成子行程。
    print(f"\n===== {label} (malicious={sorted(malicious_ids)}, flip_rate={flip_rate}) =====", flush=True)
    model_dir = os.path.join(BASE_DIR, "outputs", "models", f"attack_{label}")
    log_dir = os.path.join(BASE_DIR, "outputs", "logs", f"attack_{label}")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # --- 啟動 server（bagging + 固定 leaf_scale） ---
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
            f"--leaf_scale={LEAF_SCALE}",
        ],
        stdout=server_log, stderr=subprocess.STDOUT, cwd=BASE_DIR,
    )
    time.sleep(3)

    # ==================== 區塊 5：啟動 client ====================
    # 依序起 5 個 client 子行程，malicious_ids 裡的節點多帶 --malicious/--label_flip_rate。
    # --- 啟動全部 5 個 client，malicious_ids 裡的節點加上攻擊參數 ---
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
    # 輪詢 server 行程是否結束（逾時就強制關閉），再收掉 server 跟全部 client。
    # --- 等 server 結束或逾時、收拾所有子行程 ---
    start = time.time()
    timeout_s = 600
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
    # 讀回剛才寫出的 server.log 全文，用區塊 2 的正則逐輪抓出數字，append 進呼叫端傳入的 list。
    # --- 解析 server log：逐輪 accuracy/f1/margin、參與節點數、每節點分數 ---
    with open(server_log_path, "r", encoding="utf-8") as f:
        server_text = f.read()
    for m in ROUND_RE.finditer(server_text):
        rnd, acc, f1, mmin, mmax, mmean = m.groups()
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
    for m in CLIENT_SCORE_RE.finditer(server_text):
        rnd, client_id, cid, malicious, f1, acc, prec, recall = m.groups()
        score_rows.append({
            "config": label, "round": int(rnd), "client_id": client_id, "cid": cid,
            "malicious": malicious, "f1": float(f1), "accuracy": float(acc),
            "precision": float(prec), "recall": float(recall),
        })

    # ==================== 區塊 8：後續分析 ====================
    # 對這組剛訓練完的第 10 輪模型另外呼叫 analyze_recall_by_attack.py，算逐攻擊類型 recall。
    # --- 對第 10 輪的模型跑逐攻擊類型 recall 分析（只讀 chunk_6） ---
    round10_model = os.path.join(model_dir, f"global_model_round_{NUM_ROUNDS}.ubj")
    recall_path = os.path.join(RESULTS_DIR, f"attack_injection_{label}_recall.txt")
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
# 依序對 CONFIGS 每組呼叫 run_one()，全部跑完後把彙整出的三份資料各自寫成一份 csv。
def main():
    """依序對 CONFIGS 裡的 (no_attack, attack_c1_full) 兩組呼叫 run_one()，
    全部跑完後把彙整出的 summary_rows/participation_rows/score_rows 各自
    寫成一份 csv。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows, participation_rows, score_rows = [], [], []
    for label, malicious_ids, flip_rate in CONFIGS:
        run_one(label, malicious_ids, flip_rate, summary_rows, participation_rows, score_rows)

    summary_path = os.path.join(RESULTS_DIR, "attack_injection_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "round", "accuracy", "f1", "margin_min", "margin_max", "margin_mean",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[Info] wrote {summary_path} ({len(summary_rows)} rows)")

    participation_path = os.path.join(RESULTS_DIR, "attack_injection_participation.csv")
    with open(participation_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "round", "participating_clients", "num_clients"])
        writer.writeheader()
        writer.writerows(participation_rows)
    print(f"[Info] wrote {participation_path} ({len(participation_rows)} rows)")

    score_path = os.path.join(RESULTS_DIR, "attack_injection_client_scores.csv")
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "round", "client_id", "cid", "malicious", "f1", "accuracy", "precision", "recall",
        ])
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"[Info] wrote {score_path} ({len(score_rows)} rows)")


if __name__ == "__main__":
    main()
