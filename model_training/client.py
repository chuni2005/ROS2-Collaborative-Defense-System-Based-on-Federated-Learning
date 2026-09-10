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
import tempfile
import traceback

from preprocessing import preprocess_data, load_category_maps

TENSOR_TYPE = "xgboost-ubj"
NUM_BOOST_ROUND = 15  # 20
POSITIVE_CLASS = 1
ATTACK_SEED_BASE = 10000
PENALTY_FOR_OBSERVE = 1.5


def load_local_data(client_id, data_path, category_maps_path, eval_data_path=None):
    print(f"\n[Info] [Client {client_id}] Loading local training data: {data_path}")

    category_maps = load_category_maps(category_maps_path)
    df_train_raw = pd.read_csv(data_path, low_memory=False)
    df_train_cleaned = preprocess_data(df_train_raw, category_maps)

    label = df_train_cleaned["attack"]
    features = df_train_cleaned.drop(["attack"], axis=1)
    train_feature_names = features.columns.tolist()

    print(
        f"[Diagnostic] [Client {client_id}] Overall label distribution:\n{label.value_counts()}"
    )

    label_counts = label.value_counts()
    can_stratify = label_counts.shape[0] > 1 and label_counts.min() >= 2

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

    dup_mask = pd.concat([x_train, x_test_local]).duplicated(keep=False)
    n_dup_in_test = dup_mask.iloc[len(x_train) :].sum()
    print(
        f"[Diagnostic] [Client {client_id}] Duplicate rows between train/test "
        f"(by features): {n_dup_in_test} / {len(x_test_local)}"
    )

    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 0] = PENALTY_FOR_OBSERVE

    dtrain = xgb.DMatrix(
        x_train.values,
        label=y_train.values,
        feature_names=train_feature_names,
        weight=sample_weights,
    )
    dval = xgb.DMatrix(
        x_val.values,
        label=y_val.values,
        feature_names=train_feature_names,
    )
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

    def __init__(
        self,
        client_id,
        data_path,
        category_maps_path,
        eval_data_path=None,
        aggregation="winner",
        turn=0,
    ):
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
        ) = load_local_data(client_id, data_path, category_maps_path, eval_data_path)

        self.bst = None
        self.current_round = 0
        self.model_load_failures = 0
        self.params = {
            "objective": "binary:logistic",
            "eta": 0.3,
            "max_depth": 8,
            "eval_metric": ["logloss"],
            "tree_method": "hist",
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
            output_dir,
            f"client_{self.client_id}_turn_{self.turn}_round_{self.current_round}.ubj",
        )
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

        self._set_booster_from_parameters(ins.parameters)
        self._evaluate_global_on_local_test()

        print(f"[Info] [Client {self.client_id}] Training...")
        self.bst = xgb.train(
            self.params,
            self.dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(self.dtrain, "train"), (self.dval, "val")],
            xgb_model=self.bst,
            verbose_eval=False,
        )

        local_metrics = self._evaluate_local_test()
        self._save_model_artifact()

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
        "--category_maps_path",
        type=str,
        default="category_maps.json",
        help="Path to the shared category_maps.json produced once from the full "
        "dataset (must match the file server.py/test_model.py use).",
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
        "--turn", type=int, default=0, help="the turn of training (for save model)"
    )
    args = parser.parse_args()

    print(f"Flower Client {args.client_id} running ...")
    fl.client.start_client(
        server_address=args.server_address,
        client=XGBoostClient(
            args.client_id,
            args.data_path,
            args.category_maps_path,
            aggregation=args.aggregation,
            turn=args.turn,
        ),
    )