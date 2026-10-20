import flwr as fl
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
import argparse
import ctypes
import os
import re
import tempfile
import time
import traceback
from contextlib import contextmanager
from ctypes import wintypes

TENSOR_TYPE = "xgboost-ubj"
NUM_BOOST_ROUND = 10
POSITIVE_CLASS = 1
ATTACK_SEED_BASE = 10000


def _read_process_memory():
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        if not get_process_memory_info(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "rss": counters.WorkingSetSize,
            "peak_rss": counters.PeakWorkingSetSize,
            "private": counters.PrivateUsage,
            "private_commit": counters.PagefileUsage,
            "peak_private_commit": counters.PeakPagefileUsage,
        }

    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            values = {}
            for line in status_file:
                parts = line.split()
                if len(parts) >= 2 and parts[0].rstrip(":") in {
                    "VmRSS",
                    "VmHWM",
                    "VmSize",
                    "VmPeak",
                    "VmSwap",
                }:
                    values[parts[0].rstrip(":")] = int(parts[1]) * 1024
        return {
            "rss": values.get("VmRSS"),
            "peak_rss": values.get("VmHWM"),
            "private": None,
            "virtual": values.get("VmSize"),
            "peak_virtual": values.get("VmPeak"),
            "swap": values.get("VmSwap"),
        }
    except (OSError, ValueError):
        return {
            "rss": None,
            "peak_rss": None,
            "private": None,
            "virtual": None,
        }


def _read_system_memory():
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(memory_status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        global_memory_status_ex = kernel32.GlobalMemoryStatusEx
        global_memory_status_ex.argtypes = [ctypes.POINTER(MemoryStatusEx)]
        global_memory_status_ex.restype = wintypes.BOOL
        if not global_memory_status_ex(ctypes.byref(memory_status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "total": memory_status.ullTotalPhys,
            "available": memory_status.ullAvailPhys,
        }

    try:
        with open("/proc/meminfo", encoding="utf-8") as memory_file:
            values = {}
            for line in memory_file:
                key, _, value = line.partition(":")
                if key in {"MemTotal", "MemAvailable", "MemFree"}:
                    values[key] = int(value.split()[0]) * 1024
        return {
            "total": values.get("MemTotal"),
            "available": values.get("MemAvailable", values.get("MemFree")),
        }
    except (OSError, ValueError):
        return {"total": None, "available": None}


def _read_system_cpu_times():
    if os.name == "nt":
        class FileTime(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        def filetime_value(filetime):
            return (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime

        idle = FileTime()
        kernel = FileTime()
        user = FileTime()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_system_times = kernel32.GetSystemTimes
        get_system_times.argtypes = [
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        ]
        get_system_times.restype = wintypes.BOOL
        if not get_system_times(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return filetime_value(kernel) + filetime_value(user), filetime_value(idle)

    try:
        with open("/proc/stat", encoding="utf-8") as stat_file:
            fields = stat_file.readline().split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values[:8]), idle
    except (OSError, ValueError, IndexError):
        return None


class ResourceMonitor:
    def __init__(self):
        self._last_wall_time = None
        self._last_process_time = None
        self._last_system_cpu_times = None

    def log(self, client_id, stage):
        wall_time = time.perf_counter()
        process_time = time.process_time()
        process_cpu_pct = None
        if self._last_wall_time is not None:
            elapsed = wall_time - self._last_wall_time
            if elapsed > 0:
                process_cpu_pct = (
                    (process_time - self._last_process_time) / elapsed * 100
                )

        errors = []
        try:
            system_cpu_times = _read_system_cpu_times()
        except (OSError, AttributeError, TypeError) as error:
            system_cpu_times = None
            errors.append(f"system_cpu={error}")
        system_cpu_pct = None
        if system_cpu_times and self._last_system_cpu_times:
            total_delta = system_cpu_times[0] - self._last_system_cpu_times[0]
            idle_delta = system_cpu_times[1] - self._last_system_cpu_times[1]
            if total_delta > 0:
                system_cpu_pct = (total_delta - idle_delta) / total_delta * 100

        self._last_wall_time = wall_time
        self._last_process_time = process_time
        self._last_system_cpu_times = system_cpu_times

        try:
            process_memory = _read_process_memory()
        except (OSError, AttributeError, TypeError) as error:
            process_memory = {}
            errors.append(f"process_memory={error}")
        try:
            system_memory = _read_system_memory()
        except (OSError, AttributeError, TypeError) as error:
            system_memory = {}
            errors.append(f"system_memory={error}")

        def megabytes(value):
            return f"{value / (1024 ** 2):.1f}MB" if value is not None else "N/A"

        total_memory = system_memory.get("total")
        available_memory = system_memory.get("available")
        memory_load = (
            f"{(total_memory - available_memory) / total_memory * 100:.1f}%"
            if total_memory and available_memory is not None
            else "N/A"
        )
        process_private_commit = process_memory.get("private_commit")
        if process_private_commit is None:
            process_private_commit = process_memory.get("private")
        peak_private_commit = process_memory.get("peak_private_commit")

        def percentage(value):
            return f"{value:.1f}%" if value is not None else "N/A"

        print(
            f"[Resource] [Client {client_id}] {stage} | "
            f"RSS={megabytes(process_memory.get('rss'))} "
            f"peak_RSS={megabytes(process_memory.get('peak_rss'))} "
            f"private={megabytes(process_memory.get('private'))} "
            f"virtual={megabytes(process_memory.get('virtual'))} "
            f"peak_virtual={megabytes(process_memory.get('peak_virtual'))} "
            f"private_commit={megabytes(process_private_commit)} "
            f"peak_private_commit={megabytes(peak_private_commit)} "
            f"swap={megabytes(process_memory.get('swap'))} "
            f"system_RAM_total={megabytes(total_memory)} "
            f"system_RAM_used={memory_load} "
            f"system_RAM_available={megabytes(available_memory)} "
            f"process_CPU_single_core={percentage(process_cpu_pct)} "
            f"system_CPU_all_cores={percentage(system_cpu_pct)} "
            f"process_CPU_time={process_time:.2f}s "
            f"logical_CPUs={os.cpu_count() or 'N/A'}"
            f"{' | diagnostic_error=' + '; '.join(errors) if errors else ''}",
            flush=True,
        )


_resource_monitor = ResourceMonitor()


@contextmanager
def _resource_checkpoint(client_id, stage):
    _resource_monitor.log(client_id, f"{stage} - before")
    try:
        yield
    finally:
        _resource_monitor.log(client_id, f"{stage} - after")


def convert_type(x):
    if (isinstance(x, (int, float, np.number)) and not pd.isna(x)) and not isinstance(
        x, bool
    ):
        return x

    if pd.isna(x) or x == "":
        return ""

    try:
        num = pd.to_numeric(x)
        return num
    except Exception:
        try:
            return str(x)
        except Exception:
            return ""


def preprocess_data(df):
    feature_cols = [col for col in df.columns if col != "attack"]

    if hasattr(df, "map"):
        df[feature_cols] = df[feature_cols].map(convert_type)
    else:
        df[feature_cols] = df[feature_cols].applymap(convert_type)

    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    if "attack" in df.columns:
        attack_mapping = {
            "observe": 0,
            "metasploit SYN flood": 1,
            "nmap discovery": 1,
            "nmap SYN flood": 1,
            "ros2 node crashing": 1,
            "ros2 reconnaissance": 1,
            "ros2 reflection": 1,
        }
        df["attack"] = df["attack"].replace(attack_mapping).infer_objects(copy=False)
        df["attack"] = (
            pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)
        )

    df = df.drop(
        columns=[i for i in df.columns if "Unnamed" in i or "timestamp" in i],
        errors="ignore",
    )
    non_numeric_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns
    for col in non_numeric_cols:
        if col != "attack":
            df[col] = pd.Categorical(df[col]).codes

    df.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in df.columns]
    return df.astype(np.float32)


def load_local_data(client_id, data_path, eval_data_path=None):
    print(f"\n[Info] [Client {client_id}] Loading local training data: {data_path}")

    with _resource_checkpoint(client_id, "CSV read"):
        df_train_raw = pd.read_csv(data_path, low_memory=False)
    with _resource_checkpoint(client_id, "data preprocessing"):
        df_train_cleaned = preprocess_data(df_train_raw)

    label = df_train_cleaned["attack"]
    features = df_train_cleaned.drop(["attack"], axis=1)
    train_feature_names = features.columns.tolist()

    print(
        f"[Diagnostic] [Client {client_id}] Overall label distribution:\n{label.value_counts()}"
    )

    label_counts = label.value_counts()
    can_stratify = label_counts.shape[0] > 1 and label_counts.min() >= 2

    with _resource_checkpoint(client_id, "train/validation/test split"):
        x_train_val, x_test_local, y_train_val, y_test_local = train_test_split(
            features,
            label,
            test_size=0.2,
            random_state=client_id,
            stratify=label if can_stratify else None,
        )

        train_val_counts = y_train_val.value_counts()
        can_stratify_2 = train_val_counts.shape[0] > 1 and train_val_counts.min() >= 2

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.25,
            random_state=client_id,
            stratify=y_train_val if can_stratify_2 else None,
        )

    print(
        f"[Diagnostic] [Client {client_id}] Train label dist:\n{y_train.value_counts()}"
    )
    print(f"[Diagnostic] [Client {client_id}] Val label dist:\n{y_val.value_counts()}")
    print(
        f"[Diagnostic] [Client {client_id}] Test label dist:\n{y_test_local.value_counts()}"
    )

    with _resource_checkpoint(client_id, "duplicate-row check"):
        dup_mask = pd.concat([x_train, x_test_local]).duplicated(keep=False)
        n_dup_in_test = dup_mask.iloc[len(x_train) :].sum()
    print(
        f"[Diagnostic] [Client {client_id}] Duplicate rows between train/test "
        f"(by features): {n_dup_in_test} / {len(x_test_local)}"
    )

    with _resource_checkpoint(client_id, "DMatrix creation - train"):
        dtrain = xgb.DMatrix(
            x_train.values, label=y_train.values, feature_names=train_feature_names
        )
    with _resource_checkpoint(client_id, "DMatrix creation - validation"):
        dval = xgb.DMatrix(
            x_val.values, label=y_val.values, feature_names=train_feature_names
        )
    with _resource_checkpoint(client_id, "DMatrix creation - test"):
        dtest_local = xgb.DMatrix(
            x_test_local.values,
            label=y_test_local.values,
            feature_names=train_feature_names,
        )

    print(
        f"[Info] Loading local test data. Samples: {dtrain.num_row()}, Features: {dtrain.num_col()}"
    )

    deval_external = None
    y_eval_external = None

    return (
        dtrain,
        dval,
        dtest_local,
        deval_external,
        y_eval_external,
        len(x_train),
        len(x_test_local),
        y_test_local.values,
    )


class XGBoostClient(fl.client.Client):

    def __init__(self, client_id, data_path, eval_data_path=None, aggregation="winner", turn=0):
        self.client_id = client_id
        self.aggregation = aggregation
        self.turn = turn

        (
            self.dtrain,
            self.dval,
            self.dtest_local,
            self.deval_ext,
            self.y_ext,
            self.num_train,
            self.num_test,
            self.y_test_local,
        ) = load_local_data(client_id, data_path, eval_data_path)

        self.bst = None
        self.current_round = 0
        self.model_load_failures = 0
        self.params = {
            "objective": "binary:logistic",
            "eta": 0.1,
            "max_depth": 5,
            "eval_metric": ["logloss"],
            "tree_method": "hist",
        }

    def _load_model_from_bytes(self, model_bytes):
        if not model_bytes:
            return None

        tmp_file = None
        try:
            with _resource_checkpoint(
                self.client_id, f"model load ({len(model_bytes)} bytes)"
            ):
                with tempfile.NamedTemporaryFile(
                    "wb", suffix=".ubj", delete=False
                ) as tmp:
                    tmp.write(bytes(model_bytes))
                    tmp_file = tmp.name

                booster = xgb.Booster()
                booster.load_model(tmp_file)
                return booster

        except Exception as e:
            self.model_load_failures += 1
            print(
                f"[ERROR] [Client {self.client_id}] Model load failed "
                f"(total failures so far: {self.model_load_failures}): {e}"
            )
            traceback.print_exc()
            return None
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _serialize_model_to_bytes(self):
        if self.bst is None:
            return b""

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".ubj", delete=False) as tmp:
                tmp_file = tmp.name

            with _resource_checkpoint(self.client_id, "model serialization"):
                self.bst.save_model(tmp_file)
                with open(tmp_file, "rb") as handle:
                    model_bytes = handle.read()
            print(
                f"[Resource] [Client {self.client_id}] serialized_model_size="
                f"{len(model_bytes)} bytes",
                flush=True,
            )
            return model_bytes
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _save_model_artifact(self):
        importance = self.bst.get_score(importance_type="gain")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("Top 15 features by gain:")
        for feat, score in sorted_importance[:15]:
            print(f"  {feat}: {score:.2f}")

        print(
            f"[Info] [Client {self.client_id}] Saving model artifact for round {self.current_round}..."
        )
        if self.bst is None:
            return None

        output_dir = os.path.join(os.getcwd(), "output_models")
        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(
            output_dir, f"client_{self.client_id}_turn_{self.turn}_round_{self.current_round}.ubj"
        )
        with _resource_checkpoint(self.client_id, "model artifact save"):
            self.bst.save_model(artifact_path)
        print(f"[Info] model saved: {artifact_path}")
        return artifact_path

    def _set_booster_from_parameters(self, parameters: Parameters):
        if parameters is not None and parameters.tensors:
            model_bytes = bytes(parameters.tensors[0])
            self.bst = self._load_model_from_bytes(model_bytes)
        else:
            self.bst = None

    def _parameters_from_booster(self) -> Parameters:
        model_bytes = self._serialize_model_to_bytes()
        tensors = [model_bytes] if model_bytes else []
        return Parameters(tensors=tensors, tensor_type=TENSOR_TYPE)

    def _parameters_from_new_trees(self) -> Parameters:
        if self.bst is None:
            return Parameters(tensors=[], tensor_type=TENSOR_TYPE)

        total_rounds = self.bst.num_boosted_rounds()
        new_trees_bst = self.bst[total_rounds - NUM_BOOST_ROUND : total_rounds]

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".ubj", delete=False) as tmp:
                tmp_file = tmp.name
            with _resource_checkpoint(self.client_id, "new-tree serialization"):
                new_trees_bst.save_model(tmp_file)
                with open(tmp_file, "rb") as handle:
                    model_bytes = handle.read()
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

        return Parameters(
            tensors=[model_bytes] if model_bytes else [], tensor_type=TENSOR_TYPE
        )

    def _evaluate_global_on_local_test(self):
        print("[Info-Test] Server global model - local test")
        if self.bst is None:
            print(
                "[Warning] Global model is None, skipping evaluation on local test set."
            )
            return

        try:
            preds_prob = self.bst.predict(self.dtest_local)
            preds = (preds_prob > 0.5).astype(int)
            acc = accuracy_score(self.y_test_local, preds)
            precision = precision_score(
                self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
            )
            recall = recall_score(
                self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
            )
            f1 = f1_score(
                self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
            )
            print(
                f">>> [Info-Test] Accuracy: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}"
            )
        except Exception as e:
            print(f"[Warning] Global model evaluation failed: {e}")

    def _evaluate_local_test(self):
        print("[Info-Test] Client local model - local test")
        if self.bst is None:
            print(
                "[Warning] Local model is None, skipping evaluation on local test set."
            )
            return {
                "logloss": 1.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        try:
            preds_prob = self.bst.predict(self.dtest_local)
            preds = (preds_prob > 0.5).astype(int)
            accuracy = float(accuracy_score(self.y_test_local, preds))
            precision = float(
                precision_score(
                    self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
                )
            )
            recall = float(
                recall_score(
                    self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
                )
            )
            f1 = float(
                f1_score(
                    self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0
                )
            )

            eps = 1e-7
            preds_prob_clipped = np.clip(preds_prob, eps, 1 - eps)
            y_true = np.asarray(self.y_test_local, dtype=np.float64)
            logloss = float(
                -np.mean(
                    y_true * np.log(preds_prob_clipped)
                    + (1 - y_true) * np.log(1 - preds_prob_clipped)
                )
            )

            print(
                f"[Info-Test] Accuracy: {accuracy:.4f} Precision: {precision:.4f} "
                f"Recall: {recall:.4f} F1: {f1:.4f} LogLoss: {logloss:.4f}"
            )
            return {
                "logloss": logloss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        except Exception as e:
            print(f"[Warning] Local model evaluation failed: {e}")
            return {
                "logloss": 1.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = self._parameters_from_booster()
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        self.current_round += 1
        print(
            f"\n[Client {self.client_id}] Recieve Server Message - {self.current_round} Round Fit Start"
        )

        with _resource_checkpoint(self.client_id, "received model setup"):
            self._set_booster_from_parameters(ins.parameters)
        with _resource_checkpoint(self.client_id, "global model evaluation"):
            self._evaluate_global_on_local_test()

        print(f"[Info] [Client {self.client_id}] Training...")
        with _resource_checkpoint(self.client_id, "XGBoost training"):
            self.bst = xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(self.dtrain, "train"), (self.dval, "val")],
                xgb_model=self.bst,
                verbose_eval=False,
            )

        with _resource_checkpoint(self.client_id, "local model evaluation"):
            local_metrics = self._evaluate_local_test()
        self._save_model_artifact()

        with _resource_checkpoint(self.client_id, "fit parameter serialization"):
            if self.aggregation == "bagging":
                fit_parameters = self._parameters_from_new_trees()
            else:
                fit_parameters = self._parameters_from_booster()

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=fit_parameters,
            num_examples=self.num_train,
            metrics={
                "client_id": self.client_id,
                "accuracy": local_metrics["accuracy"],
                "precision": local_metrics["precision"],
                "recall": local_metrics["recall"],
                "f1": local_metrics["f1"],
                "logloss": local_metrics["logloss"],
                "model_load_failures": self.model_load_failures,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        with _resource_checkpoint(self.client_id, "evaluation model setup"):
            self._set_booster_from_parameters(ins.parameters)

        if self.bst is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=1.0,
                num_examples=0,
                metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            )

        with _resource_checkpoint(self.client_id, "evaluation prediction"):
            local_metrics = self._evaluate_local_test()
        print(
            f"[Info] Client {self.client_id}] local test accuracy: {local_metrics['accuracy']:.4f}"
        )
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=local_metrics["logloss"],
            num_examples=self.num_test,
            metrics={
                "accuracy": local_metrics["accuracy"],
                "precision": local_metrics["precision"],
                "recall": local_metrics["recall"],
                "f1": local_metrics["f1"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower XGBoost Client")
    parser.add_argument(
        "--cid",
        "--client_id",
        type=int,
        required=True,
        dest="client_id",
        help="Client ID",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Local training data CSV path"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080"),
        help="Flower server address",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="winner",
        choices=["winner", "bagging"],
        help="Must match the server's --aggregation. winner = send the whole cumulative "
        "model each round (default). bagging = send only this round's new trees.",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=0,
        help="the turn of training (for save model)"
    )
    args = parser.parse_args()

    print(f"Flower Client {args.client_id} running ...")
    fl.client.start_client(
        server_address=args.server_address,
        client=XGBoostClient(
            args.client_id, args.data_path, aggregation=args.aggregation, turn=args.turn
        ),
    )
