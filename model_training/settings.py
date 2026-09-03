import math
import os
import sys
import ctypes
import subprocess
import time
import shutil
import pandas as pd

NUM_CLIENTS = 5
NUM_ROUNDS = 10
TARGET_DATA = "ROSPaCe_reduced/ROSPaCe_reduced_merged.csv"
VALIDATION_DATA = f"split_data/chunk_{NUM_CLIENTS+1}.csv"
# 只給訓練結束後的最終報告用：test_natural 報整體 F1，test_rare 報逐攻擊
# 類型 recall。訓練期間的模型選擇與 ERR/LFR 決策一律只能讀 VALIDATION_DATA，
# 不得讀取這兩份。
TEST_DATA_NATURAL = "split_data/test_natural.csv"
TEST_DATA_RARE = "split_data/test_rare.csv"

SPLIT_UNIT = 1200000
SPLIT_DIR = "split_data"
SPLIT_RANDOM = True
# Fixes notes/13-progress-summary.md's reproducibility risk: split.py's
# reservoir sampling had no fixed seed, so re-running split_data() produced
# a different split every time. See notes/12b-branch-delta.md for the seed
# value and what it does/doesn't make reproducible.
SPLIT_SEED = 42

LOG_DIR = "logs"
MODEL_DIR = "model"
SPACE_CHECK = False


class SysLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class MainRunner(object):
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(self.base_dir, LOG_DIR)
        self.head = 0
        self.server_proc = None
        self.client_procs = []
        self.round_times = []
        self.num_row = self.get_data_row_num()
        self.total_rounds = max(1, math.ceil(self.num_row / SPLIT_UNIT))
        self.detect(SPACE_CHECK)
        self.init()

    def __del__(self):
        if hasattr(self, "sys_logger") and self.sys_logger is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.sys_logger.close()
            self.sys_logger = None

    def is_done(self):
        return self.head >= self.num_row

    def split_data(self):
        print(f"[Info] [{self.head}-{self.head + SPLIT_UNIT}] Splitting Data... [1/3]")
        with open(os.path.join(self.log_dir, "split.log"), "a", encoding="utf-8") as split_log:
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(self.base_dir, "split.py"),
                    f"--target_data={self._resolve_data_path()}",
                    f"--split_dir={os.path.join(self.base_dir, SPLIT_DIR)}",
                    f"--unit={SPLIT_UNIT}",
                    f"--chunk={NUM_CLIENTS + 1}", # one for server validation data
                ] + ([f"--random", f"--seed={SPLIT_SEED}"] if SPLIT_RANDOM else []),
                stdout=split_log,
                stderr=split_log,
                check=True,
            )

    def run_server(self):
        print("[Info] Running Flower Server... [2/3]")
        with open(os.path.join(self.log_dir, "server.log"), "a", encoding="utf-8") as server_log:
            self.server_proc = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(self.base_dir, "server.py"),
                    f"--model_dir={os.path.join(self.base_dir, MODEL_DIR)}",
                    f"--num_clients={NUM_CLIENTS}",
                    f"--num_rounds={NUM_ROUNDS}",
                    f"--validation_data_path={VALIDATION_DATA}",
                ],
                stdout=server_log,
                stderr=server_log,
                cwd=self.base_dir,
            )
        time.sleep(3)

    def run_clients(self):
        self.client_procs = []
        for i in range(1, NUM_CLIENTS + 1):
            print(f"[Info] Running Clients [3/3: {i}/{NUM_CLIENTS}]")
            with open(os.path.join(self.log_dir, f"client_{i}.log"), "a", encoding="utf-8") as client_log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.base_dir, "client.py"),
                        f"--client_id={i}",
                        f"--data_path={self._resolve_client_data_path(i)}",
                    ],
                    stdout=client_log,
                    stderr=client_log,
                    cwd=self.base_dir,
                )
                self.client_procs.append(proc)

    def life_check(self):
        if self.server_proc is None:
            print("\n[Info] Bruh cannot find the server process. Please ensure the server is running.")
            sys.exit(1)

        start_time = time.time()
        while True:
            time.sleep(1)
            if self.server_proc.poll() is not None:
                elapsed_seconds = time.time() - start_time
                self.round_times.append(elapsed_seconds)
                remaining_rounds = max(0, self.total_rounds - self.head // SPLIT_UNIT - 1)
                avg_round_time = sum(self.round_times) / len(self.round_times) if self.round_times else elapsed_seconds
                estimated_time = avg_round_time * remaining_rounds
                print(f"\n[Info] Round finished in {self._format_duration(elapsed_seconds)}; estimated remaining: {self._format_duration(estimated_time)}")
                print("[Info] This turn is over.")
                break

            elapsed_seconds = time.time() - start_time
            print(f"\r[Info] [{self._format_duration(elapsed_seconds)}] FL doing... ", end="", flush=True)

    def shut_down(self):
        self._stop_process(self.server_proc)
        self.server_proc = None
        for proc in self.client_procs:
            self._stop_process(proc)
        self.client_procs = []

    def _stop_process(self, proc):
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def next(self):
        self.head += SPLIT_UNIT

    def _resolve_data_path(self):
        if not TARGET_DATA:
            print("[Warning] Hey, man! TARGET_DATA is not specified.")
            sys.exit(1)
        return TARGET_DATA if os.path.isabs(TARGET_DATA) else os.path.abspath(os.path.join(self.base_dir, TARGET_DATA))

    def _resolve_client_data_path(self, client_id):
        candidate = os.path.join(self.base_dir, SPLIT_DIR, f"chunk_{client_id}.csv")
        if os.path.exists(candidate):
            return candidate
        return self._resolve_data_path()

    def get_data_row_num(self):
        data_path = self._resolve_data_path()
        if not os.path.exists(data_path):
            print("[Warning] Bruh your source data file is not found.")
            sys.exit(1)

        total_rows = 0
        chunk_size = 50000
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=[0]):
            total_rows += len(chunk)
        return total_rows

    def detect(self, file_check: bool = True):
        # self.admin_check()

        if SPLIT_UNIT % NUM_CLIENTS != 0:
            print("[Warning] Bruh your SPLIT_UNIT must be divisible by NUM_CLIENTS!")
            sys.exit(1)

        if not file_check:
            return

        data_path = self._resolve_data_path()
        file_size_bytes = os.path.getsize(data_path)
        file_size_gb = file_size_bytes / (1024**3)
        split_size_gb = SPLIT_UNIT * file_size_gb / self.num_row
        required_space_gb = file_size_gb * 2.0 + split_size_gb + 1
        free_gb = shutil.disk_usage(self.base_dir).free / (1024**3)

        if free_gb < required_space_gb:
            print(f"[Warning] Bruh your disk is too small, it should larger than {required_space_gb} GB.")
            sys.exit(1)

    def init(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.sys_logger = SysLogger(os.path.join(self.log_dir, "run.log"))
        sys.stdout = self.sys_logger
        sys.stderr = self.sys_logger

    def admin_check(self):
        if os.name != "nt":
            return
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            is_admin = False
        if not is_admin:
            print("[Warning] Hey, this runner is not running as Administrator. Proceeding without elevation.")

    def _format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"