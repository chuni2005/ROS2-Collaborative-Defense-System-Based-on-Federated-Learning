"""量每個誠實節點被拿掉之後，全域模型的表現變化多少（留一法 impact）。

原理：訓練一次「全部節點都在」的模型（候選 A），再訓練幾次「只拿掉一個
節點」的模型（候選 B_1~B_5），比較 A 跟每個 B_i 的表現差多少。如果某個
節點的資料或行為特別可疑，拿掉它應該會讓模型明顯變好；如果拿掉任何節點的
影響都差不多、都很小，代表目前沒有哪個節點特別突出。

輸入：這支腳本不接受參數，設定寫死在檔案開頭常數（LEAF_SCALE、
ALL_CLIENTS 等）；實際輸入是磁碟上既有的 split_data/chunk_1.csv~chunk_6.csv
（5 個節點的訓練資料 + 1 份伺服器驗證資料），不會重新切分。
輸出：results/loo_impact_summary.csv（六組設定 × 10 輪的逐輪
accuracy/f1/margin），以及終端機上印出的一張 impact 表（config／F1／
Impact(A-Bi)）。

怎麼做：六組設定（候選 A 全部 5 個節點；候選 B_i 排除節點 i，其餘 4 個，
--num_clients=4 讓 Flower 內部的連線門檻對得上實際連上的節點數）分別各自
跑一次完整的 10 輪聯邦 bagging 訓練（真的啟動 server.py 加上對應數量的
client.py 子行程，不是模擬）。A 跟每個 B_i 都用磁碟上既有的同一份
chunk_1..6.csv、同一個 leaf_scale=0.5，排除掉「資料切分不同」或「縮放
係數不同」這兩個混淆變因，讓 A 跟 B_i 之間唯一的差異就是「有沒有那一個
節點的資料跟樹」。全部跑完之後，取每組第 10 輪的 F1：Impact_i = F1_A −
F1_Bi，正值代表拿掉這個節點讓模型變差，負值代表拿掉反而變好。

為什麼需要它：這是 ERR/LFR 的核心動作（拿掉一個節點、重新聚合、比較跟
「全部保留」的差異）已經在跑，但目前只是量測，不是拿去決定要不要真的剔除
某個節點——量出來的差異是拿去跟「雜訊底噪」比較的：如果連拿掉一個誠實
節點都會造成不小的變化，代表這套量測方法本身雜訊很大，拿它來判斷「這個
節點可疑」會不準。要變成真正的 ERR/LFR，還缺：(1) 每一輪即時算 impact，
不是跑完整個 10 輪才算一次；(2) 依 impact 排序，選出最大的 c 個節點；
(3) 把這 c 個節點從這一輪的聚合裡真的排除掉，不是像現在這樣拿掉之後重跑
一次獨立的訓練。

這是重做原本的留一法量測——那一次直接呼叫 Flower 官方的 aggregate()，重現
了跟 aggregate_bagging_verified() 同樣的 bug（只取到每個模型裡從未更新過
的舊樹），量到的 impact 是合併邏輯壞掉產生的雜訊，不是真實訊號。這支腳本
在修正後的 pipeline 上重新量一次，符號慣例（Impact_i = F1_A − F1_Bi）
跟原本一致。

完整調查過程見 notes/12-baseline.md、notes/13a-bagging-baseline.md。
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
