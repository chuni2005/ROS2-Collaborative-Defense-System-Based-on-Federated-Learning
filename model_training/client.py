# 拆給server val

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
import os
import re
import tempfile
import traceback


TENSOR_TYPE = "xgboost-ubj"
NUM_BOOST_ROUND = 10  # trees added per federated round; also used to slice
                      # out "this round's new trees" in bagging mode.

# Positive class for precision/recall/F1 is 1 ("attack") -- must match
# server.py's POSITIVE_CLASS. See notes/11-dataset-check.md.
POSITIVE_CLASS = 1


def convert_type(x):
    if (isinstance(x, (int, float, np.number)) and not pd.isna(x)) and not isinstance(x, bool):
        return x

    if pd.isna(x) or x == '':
        return ''

    try:
        num = pd.to_numeric(x)
        return num
    except Exception:
        try:
            return str(x)
        except Exception:
            return ''


def preprocess_data(df):
    feature_cols = [col for col in df.columns if col != 'attack']

    if hasattr(df, 'map'):
        df[feature_cols] = df[feature_cols].map(convert_type)
    else:
        df[feature_cols] = df[feature_cols].applymap(convert_type)

    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    if 'attack' in df.columns:
        attack_mapping = {
            'observe': 0, 'metasploit SYN flood': 1, 'nmap discovery': 1,
            'nmap SYN flood': 1, 'ros2 node crashing': 1,
            'ros2 reconnaissance': 1, 'ros2 reflection': 1
        }
        df['attack'] = df['attack'].replace(attack_mapping).infer_objects(copy=False)
        df['attack'] = pd.to_numeric(df['attack'], errors='coerce').fillna(0).astype(int)

    df = df.drop(columns=[i for i in df.columns if "Unnamed" in i or "timestamp" in i], errors='ignore')

    non_numeric_cols = df.select_dtypes(exclude=[np.number, 'bool']).columns
    for col in non_numeric_cols:
        if col != 'attack':
            df[col] = pd.Categorical(df[col]).codes

    df.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in df.columns]
    return df.astype(np.float32)


def load_local_data(client_id, data_path, eval_data_path=None):
    print(f"\n[Info] [Client {client_id}] Loading local training data: {data_path}")

    df_train_raw = pd.read_csv(data_path, low_memory=False)
    df_train_cleaned = preprocess_data(df_train_raw)

    label = df_train_cleaned['attack']
    features = df_train_cleaned.drop(['attack'], axis=1)
    train_feature_names = features.columns.tolist()

    print(f"[Diagnostic] [Client {client_id}] Overall label distribution:\n{label.value_counts()}")

    label_counts = label.value_counts()
    can_stratify = label_counts.shape[0] > 1 and label_counts.min() >= 2

    x_train_val, x_test_local, y_train_val, y_test_local = train_test_split(
        features, label, test_size=0.2, random_state=client_id,
        stratify=label if can_stratify else None
    )

    train_val_counts = y_train_val.value_counts()
    can_stratify_2 = train_val_counts.shape[0] > 1 and train_val_counts.min() >= 2

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=client_id,
        stratify=y_train_val if can_stratify_2 else None
    )

    print(f"[Diagnostic] [Client {client_id}] Train label dist:\n{y_train.value_counts()}")
    print(f"[Diagnostic] [Client {client_id}] Val label dist:\n{y_val.value_counts()}")
    print(f"[Diagnostic] [Client {client_id}] Test label dist:\n{y_test_local.value_counts()}")

    dup_mask = pd.concat([x_train, x_test_local]).duplicated(keep=False)
    n_dup_in_test = dup_mask.iloc[len(x_train):].sum()
    print(
        f"[Diagnostic] [Client {client_id}] Duplicate rows between train/test "
        f"(by features): {n_dup_in_test} / {len(x_test_local)}"
    )

    dtrain = xgb.DMatrix(x_train.values, label=y_train.values, feature_names=train_feature_names)
    dval = xgb.DMatrix(x_val.values, label=y_val.values, feature_names=train_feature_names)
    dtest_local = xgb.DMatrix(x_test_local.values, label=y_test_local.values, feature_names=train_feature_names)

    print(f"[Info] Loading local test data. Samples: {dtrain.num_row()}, Features: {dtrain.num_col()}")

    deval_external = None
    y_eval_external = None

    return dtrain, dval, dtest_local, deval_external, y_eval_external, len(x_train), len(x_test_local), y_test_local.values


class XGBoostClient(fl.client.Client):

    def __init__(self, client_id, data_path, eval_data_path=None, aggregation="winner"):
        self.client_id = client_id
        self.aggregation = aggregation

        self.dtrain, self.dval, self.dtest_local, self.deval_ext, self.y_ext, \
            self.num_train, self.num_test, self.y_test_local = \
            load_local_data(client_id, data_path, eval_data_path)

        self.bst = None
        self.current_round = 0
        self.model_load_failures = 0
        self.params = {
            "objective": "binary:logistic", "eta": 0.1, "max_depth": 5,
            "eval_metric": ["logloss"], "tree_method": "hist"
        }

    def _load_model_from_bytes(self, model_bytes):
        if not model_bytes:
            return None

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".ubj", delete=False) as tmp:
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

            self.bst.save_model(tmp_file)
            with open(tmp_file, "rb") as handle:
                return handle.read()
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _save_model_artifact(self):
        importance = self.bst.get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("Top 15 features by gain:")
        for feat, score in sorted_importance[:15]:
            print(f"  {feat}: {score:.2f}")

        print(f"[Info] [Client {self.client_id}] Saving model artifact for round {self.current_round}...")
        if self.bst is None:
            return None

        output_dir = os.path.join(os.getcwd(), "output_models")
        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, f"client_{self.client_id}_round_{self.current_round}.ubj")
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

    # Bagging mode only: send just this round's newly-added trees, not the
    # whole cumulative model. Required for server-side bagging aggregation
    # to work correctly -- see notes/00-findings.md finding 21 and
    # notes/13a-bagging-baseline.md for why sending the whole model breaks
    # Flower's aggregate(). Slicing by num_boosted_rounds() is the same
    # technique Flower's own bagging example uses (client_app.py's
    # _local_boost()), verified empirically to correctly isolate the delta
    # even when continuing from a previous round's model (xgb_model=...).
    def _parameters_from_new_trees(self) -> Parameters:
        if self.bst is None:
            return Parameters(tensors=[], tensor_type=TENSOR_TYPE)

        total_rounds = self.bst.num_boosted_rounds()
        new_trees_bst = self.bst[total_rounds - NUM_BOOST_ROUND : total_rounds]

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile("wb", suffix=".ubj", delete=False) as tmp:
                tmp_file = tmp.name
            new_trees_bst.save_model(tmp_file)
            with open(tmp_file, "rb") as handle:
                model_bytes = handle.read()
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

        return Parameters(tensors=[model_bytes] if model_bytes else [], tensor_type=TENSOR_TYPE)

    # Verified against real data in notes/12-baseline.md.
    def _evaluate_global_on_local_test(self):
        print("[Info-Test] Server global model - local test")
        if self.bst is None:
            print("[Warning] Global model is None, skipping evaluation on local test set.")
            return

        try:
            preds_prob = self.bst.predict(self.dtest_local)
            preds = (preds_prob > 0.5).astype(int)
            acc = accuracy_score(self.y_test_local, preds)
            precision = precision_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0)
            recall = recall_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0)
            f1 = f1_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0)
            print(f">>> [Info-Test] Accuracy: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
        except Exception as e:
            print(f"[Warning] Global model evaluation failed: {e}")

    # Verified against real data in notes/12-baseline.md.
    # zero_division=0: precision/recall/F1 come back as 0.0 rather than
    # raising or warning when a client's local test split (test_size=0.2 of
    # its own chunk) happens to contain only one label -- not exercised
    # against real data yet, so this default hasn't been checked against
    # what our actual chunk sizes produce.
    def _evaluate_local_test(self):
        print("[Info-Test] Client local model - local test")
        if self.bst is None:
            print("[Warning] Local model is None, skipping evaluation on local test set.")
            return {"logloss": 1.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        try:
            preds_prob = self.bst.predict(self.dtest_local)
            preds = (preds_prob > 0.5).astype(int)
            accuracy = float(accuracy_score(self.y_test_local, preds))
            precision = float(precision_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0))
            recall = float(recall_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0))
            f1 = float(f1_score(self.y_test_local, preds, pos_label=POSITIVE_CLASS, zero_division=0))

            eps = 1e-7
            preds_prob_clipped = np.clip(preds_prob, eps, 1 - eps)
            y_true = np.asarray(self.y_test_local, dtype=np.float64)
            logloss = float(-np.mean(
                y_true * np.log(preds_prob_clipped) + (1 - y_true) * np.log(1 - preds_prob_clipped)
            ))

            print(
                f"[Info-Test] Accuracy: {accuracy:.4f} Precision: {precision:.4f} "
                f"Recall: {recall:.4f} F1: {f1:.4f} LogLoss: {logloss:.4f}"
            )
            return {
                "logloss": logloss, "accuracy": accuracy,
                "precision": precision, "recall": recall, "f1": f1,
            }
        except Exception as e:
            print(f"[Warning] Local model evaluation failed: {e}")
            return {"logloss": 1.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = self._parameters_from_booster()
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        self.current_round += 1
        print(f"\n[Client {self.client_id}] Recieve Server Message - {self.current_round} Round Fit Start")

        self._set_booster_from_parameters(ins.parameters)
        self._evaluate_global_on_local_test()

        print(f"[Info] [Client {self.client_id}] Training...")
        self.bst = xgb.train(
            self.params, self.dtrain, num_boost_round=NUM_BOOST_ROUND,
            evals=[(self.dtrain, "train"), (self.dval, "val")],
            xgb_model=self.bst, verbose_eval=False
        )

        local_metrics = self._evaluate_local_test()
        self._save_model_artifact()

        # winner mode sends the whole cumulative model (server picks one
        # winner's full model as next round's starting point). bagging mode
        # sends only this round's new trees -- see _parameters_from_new_trees.
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
        self._set_booster_from_parameters(ins.parameters)

        if self.bst is None:
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=1.0,
                num_examples=0,
                metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            )

        local_metrics = self._evaluate_local_test()
        print(f"[Info] Client {self.client_id}] local test accuracy: {local_metrics['accuracy']:.4f}")
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
    parser.add_argument("--cid", "--client_id", type=int, required=True, dest="client_id", help="Client ID")
    parser.add_argument("--data_path", type=str, required=True, help="Local training data CSV path")
    parser.add_argument("--server_address", type=str, default=os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080"), help="Flower server address")
    parser.add_argument("--aggregation", type=str, default="winner", choices=["winner", "bagging"],
                        help="Must match the server's --aggregation. winner = send the whole cumulative "
                             "model each round (default). bagging = send only this round's new trees.")
    args = parser.parse_args()

    print(f"Flower Client {args.client_id} running ...")
    fl.client.start_client(
        server_address=args.server_address,
        client=XGBoostClient(args.client_id, args.data_path, aggregation=args.aggregation)
    )