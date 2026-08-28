"""Sweep leaf_scale in {1/5, 1/3, 1/2, 1} to fix the margin-explosion problem
found in notes/13a-bagging-baseline.md (5 clients' independent corrections
are summed, not averaged, each round -- see server.py's
scale_leaf_values()/XGBoostBaggingStrategy docstrings for the mechanism).

Deliberately does NOT call split.py: split_data/chunk_*.csv already exists
from a previous run, and split.py's chunking has no fixed random seed
(notes/12-baseline.md), so re-splitting between sweep values would confound
"effect of leaf_scale" with "effect of a different data split". All four
sweep values below train/evaluate on the exact same chunk_1..6.csv already
on disk.

Each config gets its own model_dir (so initialize_parameters() doesn't
resume from a previous sweep value's model) and its own log dir under
logs/leaf_scale_<label>/. Per-round accuracy/F1/margin are parsed out of
the server's own stdout ("[Info] Round N bagging-merged ..." lines, see
server.py) into results/leaf_scale_summary.csv; round-10 per-attack-type
recall is captured via analyze_recall_by_attack.py into
results/leaf_scale_<label>_recall.txt.
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

# (label used in dir/file names, actual w passed to --leaf_scale)
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
    # fresh model_dir per sweep value -- initialize_parameters() would
    # otherwise resume from a previous value's global_model_latest.ubj
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
