import argparse
import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics import precision_score, recall_score, f1_score

# Positive class for precision/recall/F1 is 1 ("attack") -- ROSPaCe's attack
# samples are the MAJORITY class (~78%), the reverse of typical IDS datasets.
# Precision here = "of the models's attack predictions, how many were really
# attacks"; recall = "of the real attacks, how many did it catch". See
# notes/11-dataset-check.md and notes/00-findings.md finding 17.
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


class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None):
        self.num_clients = num_clients
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.latest_model_path = os.path.join(self.model_dir, "global_model_latest.ubj")

        if not val_data_path or not os.path.exists(val_data_path):
            raise FileNotFoundError(
                f"Server-side validation data not found at {val_data_path!r}. "
                "Refusing to start: this system does not fall back to trusting "
                "client-reported metrics for model selection."
            )

        print(f"[Info] Loading server-side validation data from {val_data_path}...")
        df_val = pd.read_csv(val_data_path)
        df_val = preprocess_data(df_val)
        X_val = df_val.iloc[:, :-1]
        y_val = df_val.iloc[:, -1]
        self.dval = xgb.DMatrix(X_val, label=y_val)
        self.y_true = y_val.values

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
        )

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        if os.path.exists(self.latest_model_path):
            with open(self.latest_model_path, "rb") as f:
                model_bytes = f.read()

            if model_bytes:
                print(
                    f"[Info] Found existing model at {self.latest_model_path}, "
                    f"resuming training from it instead of starting from scratch."
                )
                return Parameters(tensors=[model_bytes], tensor_type="xgboost-ubj")

        print(
            f"[Info] No existing model found at {self.latest_model_path}, "
            f"starting from scratch (will request initial parameters from a client)."
        )
        return None

    def _extract_payload(self, parameters: Optional[Parameters]) -> Optional[bytes]:
        if parameters is None or not getattr(parameters, "tensors", None):
            return None
        if not parameters.tensors:
            return None
        return bytes(parameters.tensors[0])

    # Verified against real data in notes/12-baseline.md (two full 10-round
    # runs, baseline_run1/run2). zero_division=0 means precision/recall/F1
    # come back as 0.0 (not a warning or exception) when a class is entirely
    # absent from y_true or
    # from the predictions, e.g. a validation chunk or client test split that
    # happens to contain only one label. Not exercised against real data yet.
    def _evaluate_model_on_server(self, model_bytes: bytes) -> Dict[str, float]:
        zero_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        if self.dval is None:
            return zero_metrics
        try:
            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            preds = bst.predict(self.dval)

            preds_binary = [1 if p > 0.5 else 0 for p in preds]
            correct = sum(1 for p, y in zip(preds_binary, self.y_true) if p == y)
            accuracy = correct / len(self.y_true)
            precision = precision_score(self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)
            recall = recall_score(self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)
            f1 = f1_score(self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)
            return {
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        except Exception as e:
            print(f"[Error] Failed to evaluate model on server: {e}")
            return zero_metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}")

        payloads = []
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if payload:
                server_metrics = self._evaluate_model_on_server(payload)
                # reported_client_id is ONLY for logging/identification in this
                # experiment (all 5 clients are honest here) -- it is NOT used
                # for scoring or trust decisions, so it does not reintroduce the
                # self-report fallback removed in commit ff82d04. A malicious
                # client could still lie about this field; if task 13 needs the
                # mapping to be trustworthy under attack, it can't rely on this.
                reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
                print(
                    f"[Server Eval] Client {client_proxy.cid} (client_id={reported_client_id}) "
                    f"Accuracy: {server_metrics['accuracy']:.4f} "
                    f"Precision: {server_metrics['precision']:.4f} "
                    f"Recall: {server_metrics['recall']:.4f} "
                    f"F1: {server_metrics['f1']:.4f}"
                )
                payloads.append((payload, server_metrics, client_proxy.cid, reported_client_id))

        # Ranking key switched from accuracy to F1 -- see
        # notes/11-dataset-check.md "誠實模型的 accuracy 基準線" and
        # notes/12-baseline.md (verified: judgment gap vs. the "always
        # predict attack" floor is ~19.6 F1 points, well above noise).
        best_payload, best_metrics, best_cid, best_client_id = max(payloads, key=lambda item: item[1]["f1"])

        model_path = os.path.join(self.model_dir, f"global_model_round_{server_round}.ubj")
        with open(model_path, "wb") as f: f.write(best_payload)
        with open(self.latest_model_path, "wb") as f: f.write(best_payload)

        print(f"[Info] Round {server_round} kept the highest-F1 model (client {best_cid}, "
              f"client_id={best_client_id}, accuracy={best_metrics['accuracy']:.4f}, "
              f"f1={best_metrics['f1']:.4f}) and saved it to {model_path}")

        aggregated_parameters = Parameters(tensors=[best_payload], tensor_type="xgboost-ubj")
        return aggregated_parameters, {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
        }

    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
    #     failures: List[BaseException],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:

    #     if not results:
    #         print(f"Round {server_round} has no evaluate results to aggregate.")
    #         return None, {}

    #     if failures:
    #         print(f"[Warning] Round {server_round} had {len(failures)} evaluate failure(s): {failures}")

    #     accuracies = [float(r.metrics.get("accuracy", 0.0)) * r.num_examples for _, r in results]
    #     losses = [float(r.loss) * r.num_examples for _, r in results]
    #     examples = [r.num_examples for _, r in results]

    #     total_examples = sum(examples)
    #     if total_examples == 0:
    #         return None, {}

    #     weighted_accuracy = sum(accuracies) / total_examples
    #     weighted_loss = sum(losses) / total_examples

    #     per_client_acc = {
    #         client_proxy.cid: float(r.metrics.get("accuracy", 0.0))
    #         for client_proxy, r in results
    #     }
    #     print(f"[Diagnostic] Round {server_round} per-client accuracy: {per_client_acc}")
    #     print(f"[Info] Round {server_round} global validation accuracy: {weighted_accuracy:.4f}, "
    #           f"loss: {weighted_loss:.4f}\n")

    #     return weighted_loss, {"accuracy": weighted_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Flower server with XGBoost strategy.")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save the trained model.")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds to train the model.")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients to train the model.")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080", help="Flower server address")
    parser.add_argument("--validation_data_path", type=str, default="split_data/chunk_6.csv", help="Path to the validation data CSV file.")
    args = parser.parse_args()

    strategy = XGBoostStrategy(
        model_dir=args.model_dir,
        num_clients=args.num_clients,
        val_data_path=args.validation_data_path
    )

    print("[Info] Flower Server (XGBoost) is starting...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )