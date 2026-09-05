import math
import os
import socket
import sys
import subprocess
import time
import shutil

from split import Chunk, RandomStrategy, Splitter, StratifiedStrategy, SequentialStrategy

NUM_CLIENTS = 7
NUM_ROUNDS = 10

TARGET_DATA = "../ROSPaCe_complete/ROSPaCe_complete_noperiodicity.csv"

TEST_DIR = "test-data"
TEST_DATA = f"{TEST_DIR}/test.csv"
TEST_RATIO = 0.1  #  ratio of every attack class

VAL_DIR = "val-data"
VALIDATION_DATA = f"{VAL_DIR}/val.csv"
VAL_RATIO = 0.005

SPLIT_DIR = "split-data"
CLIENT_STRATEGY = StratifiedStrategy
SPLIT_UNIT = 1000  # per chunks  # ss=1000
RANDOM_SEED = None

LOG_DIR = "logs"
MODEL_DIR = "model"

SERVER_ADDRESS = "127.0.0.1:8080"


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
        self.server_proc = None
        self.client_procs = []
        self.round_times = []
        self.turn = 1
        self._init()

        self.splitter = Splitter(
            src_path=self._resolve_data_path(),
            tmp_dir=os.path.join(self.base_dir, "tmp"),
            output_dir=os.path.join(self.base_dir, SPLIT_DIR),
            chart_dir=os.path.join(self.base_dir, "img"),
            chunk_size=SPLIT_UNIT,
            chunk_num=NUM_CLIENTS,
            strategy=CLIENT_STRATEGY(),
        )

        # Split Test Data
        self.splitter.split_ratio_data(
            ratio=TEST_RATIO,
            output_path=os.path.join(self.base_dir, TEST_DATA),
            random_seed=RANDOM_SEED,
        )
        self.splitter.plot_attack_labels(TEST_DATA, f"Test Distributed")

        rounds = self.splitter.it.row_num / (SPLIT_UNIT * NUM_CLIENTS)
        self.total_rounds = max(1, math.ceil(rounds))

    def __del__(self):
        self.shut_down()
        if hasattr(self, "sys_logger") and self.sys_logger is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.sys_logger.close()
            self.sys_logger = None

    def is_done(self):
        return not self.splitter.it.records

    def split_data(self):
        print(f"[Runner] Splitting Validation Data ... [1/4]")
        self.splitter.split_ratio_data(
            ratio=VAL_RATIO,
            output_path=os.path.join(self.base_dir, VALIDATION_DATA),
            random_seed=RANDOM_SEED,
        )
        print(f"[Runner] Splitting Chunk Data (clients)... [2/4]")
        self.splitter.plot_attack_labels(
            VALIDATION_DATA, f"Vaidation Distributed {self.turn} Turn"
        )
        self._split_round_client_data()

    def _split_round_client_data(self):
        split_dir = os.path.join(self.base_dir, SPLIT_DIR)
        self.splitter.chunk = Chunk(
            chunk_data_dir=split_dir,
            chunk_num=NUM_CLIENTS,
            chunk_size=SPLIT_UNIT,
        )

        self.splitter.set_strategy(CLIENT_STRATEGY())
        self.splitter.split_to_chunks()
        # img
        for i in range(NUM_CLIENTS):
            chunk_csv_path = os.path.join(split_dir, f"chunk_{i}.csv")
            self.splitter.plot_attack_labels(
                chunk_csv_path, f"Chunk {i} in {self.turn} Turn"
            )

    def run_server(self):
        print("[Runner] Running Flower Server... [3/4]")
        with open(
            os.path.join(self.log_dir, "server.log"), "a", encoding="utf-8"
        ) as server_log:
            self.server_proc = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(self.base_dir, "server.py"),
                    f"--model_dir={os.path.join(self.base_dir, MODEL_DIR)}",
                    f"--num_clients={NUM_CLIENTS}",
                    f"--num_rounds={NUM_ROUNDS}",
                    f"--validation_data_path={VALIDATION_DATA}",
                    f"--server_address={SERVER_ADDRESS}",
                ],
                stdout=server_log,
                stderr=server_log,
                cwd=self.base_dir,
            )
        server_address = SERVER_ADDRESS.split(":")
        self._wait_for_server_ready(server_address[0], int(server_address[1]))

    def _wait_for_server_ready(self, host, port, timeout=120):
        start = time.time()
        while time.time() - start < timeout:
            if self.server_proc.poll() is not None:
                raise RuntimeError(
                    "[Runner] Server process died before becoming ready."
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) == 0:
                    print("[Runner] Server is ready.")
                    return
            time.sleep(0.5)
        raise TimeoutError("[Runner] Server did not become ready in time.")

    def run_clients(self):
        self.client_procs = []
        for i in range(NUM_CLIENTS):
            print(f"[Runner] Running Clients [4/4: {i+1}/{NUM_CLIENTS}]")
            with open(
                os.path.join(self.log_dir, f"client_{i}.log"), "a", encoding="utf-8"
            ) as client_log:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.base_dir, "client.py"),
                        f"--client_id={i}",
                        f"--data_path={SPLIT_DIR}/chunk_{i}.csv",
                    ],
                    stdout=client_log,
                    stderr=client_log,
                    cwd=self.base_dir,
                )
                self.client_procs.append(proc)

    def life_check(self):
        if self.server_proc is None:
            print(
                "\n[Runner] Bruh cannot find the server process. Please ensure the server is running."
            )
            sys.exit(1)

        start_time = time.time()
        while True:
            time.sleep(1)
            if self.server_proc.poll() is not None:
                elapsed_seconds = time.time() - start_time
                self.round_times.append(elapsed_seconds)

                remaining_rounds = max(
                    0, math.ceil(self.splitter.it.row_num / (SPLIT_UNIT * NUM_CLIENTS))
                )
                avg_round_time = (
                    sum(self.round_times) / len(self.round_times)
                    if self.round_times
                    else elapsed_seconds
                )
                estimated_time = avg_round_time * remaining_rounds
                print(
                    f"\n[Runner] Round finished in {self._format_duration(elapsed_seconds)}; estimated remaining: {self._format_duration(estimated_time)}"
                )
                print("[Runner] This turn is over.")
                break

            elapsed_seconds = time.time() - start_time
            print(
                f"\r[Runner] [{self._format_duration(elapsed_seconds)}] FL doing... ",
                end="",
                flush=True,
            )

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

    def _resolve_data_path(self):
        if not TARGET_DATA:
            print("[Warning] Hey, man! TARGET_DATA is not specified.")
            sys.exit(1)
        return (
            TARGET_DATA
            if os.path.isabs(TARGET_DATA)
            else os.path.abspath(os.path.join(self.base_dir, TARGET_DATA))
        )

    def _init(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.sys_logger = SysLogger(os.path.join(self.log_dir, "run.log"))
        sys.stdout = self.sys_logger
        sys.stderr = self.sys_logger

    def _format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
