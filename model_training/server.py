import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics import precision_score, recall_score, f1_score
import json

from preprocessing import preprocess_data, load_category_maps

# precision/recall/F1 positive calss of attack lobel is 1
POSITIVE_CLASS = 1


def scale_leaf_values(model_json: bytes, w: float) -> bytes:
    model = json.loads(bytearray(model_json))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    for tree in trees:
        left_children = tree["left_children"]
        base_weights = tree["base_weights"]
        split_conditions = tree["split_conditions"]
        for i, left in enumerate(left_children):
            if left == -1:  # is leaf node
                base_weights[i] = base_weights[i] * w
                split_conditions[i] = split_conditions[i] * w
    return bytes(json.dumps(model), "utf-8")


def aggregate_bagging_verified(
    bst_prev_org: Optional[bytes], bst_curr_org: bytes
) -> bytes:
    if not bst_prev_org:
        return bst_curr_org

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    gbtree_prev = bst_prev["learner"]["gradient_booster"]["model"]
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    tree_num_prev = int(gbtree_prev["gbtree_model_param"]["num_trees"])
    num_new_trees = len(trees_curr)

    # insert trees behind the `bst_prev_org`
    gbtree_prev["gbtree_model_param"]["num_trees"] = str(tree_num_prev + num_new_trees)
    iteration_indptr = gbtree_prev["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + num_new_trees)

    for tree_count in range(num_new_trees):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        gbtree_prev["trees"].append(trees_curr[tree_count])
        gbtree_prev["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model_dir: str,
        num_clients: int,
        val_data_path: Optional[str] = None,
        category_maps_path: str = "category_maps.json",
    ):
        self.num_clients = num_clients
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.latest_model_path = os.path.join(self.model_dir, "global_model_latest.ubj")

        self.category_maps = load_category_maps(category_maps_path)
        self._load_eval_dataset(val_data_path)

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
        )

    def _load_eval_dataset(self, data_path: Optional[str]):
        if data_path and os.path.exists(data_path):
            print(f"[Info] Loading server-side validation data from {data_path}...")
            df_val = pd.read_csv(data_path, low_memory=False)
            df_val = preprocess_data(df_val, self.category_maps)
            X_val = df_val.iloc[:, :-1]
            y_val = df_val.iloc[:, -1]
            self.dval = xgb.QuantileDMatrix(X_val, label=y_val)
            self.y_true = y_val.values
        else:
            raise FileNotFoundError(
                f"Server-side eval data path given but not found: {data_path!r}"
            )

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
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

    def _evaluate_model_on_server(self, model_bytes: bytes) -> Dict[str, float]:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "margin_min": 0.0,
            "margin_max": 0.0,
            "margin_mean": 0.0,
            "logloss": 1.0,
        }
        if self.dval is None:
            return 0.0
        try:
            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            preds = bst.predict(self.dval)
            margin = bst.predict(self.dval, output_margin=True)
            preds_binary = [1 if p > 0.5 else 0 for p in preds]
            correct = sum(1 for p, y in zip(preds_binary, self.y_true) if p == y)
            metrics["accuracy"] = correct / len(self.y_true)
            metrics["precision"] = precision_score(
                self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0
            )
            metrics["recall"] = recall_score(
                self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0
            )
            metrics["f1"] = f1_score(
                self.y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0
            )

            metrics["margin_min"] = float(np.min(margin))
            metrics["margin_max"] = float(np.max(margin))
            metrics["margin_mean"] = float(np.mean(margin))

            eps = 1e-7
            preds_clipped = np.clip(preds, eps, 1 - eps)
            y_true_arr = np.asarray(self.y_true, dtype=np.float64)
            metrics["logloss"] = float(
                -np.mean(
                    y_true_arr * np.log(preds_clipped)
                    + (1 - y_true_arr) * np.log(1 - preds_clipped)
                )
            )

        except Exception as e:
            print(f"[Error] Failed to evaluate model on server: {e}")
        return metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # choose the best F1

        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(
                f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}"
            )

        payloads = []
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if payload:
                server_metrics = self._evaluate_model_on_server(payload)
                reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
                print(
                    f"[Info] Server Eval Client {client_proxy.cid} (client_id={reported_client_id}) "
                    f"Accuracy: {server_metrics['accuracy']:.4f} "
                    f"Precision: {server_metrics['precision']:.4f} "
                    f"Recall: {server_metrics['recall']:.4f} "
                    f"F1: {server_metrics['f1']:.4f}"
                )
                payloads.append(
                    (payload, server_metrics, client_proxy.cid, reported_client_id)
                )
        best_payload, best_metrics, best_cid, best_client_id = max(
            payloads, key=lambda item: item[1]["f1"]
        )

        model_path = os.path.join(
            self.model_dir, f"global_model_round_{server_round}.ubj"
        )
        with open(model_path, "wb") as f:
            f.write(best_payload)
        with open(self.latest_model_path, "wb") as f:
            f.write(best_payload)

        print(
            f"[Info] Round {server_round} kept the highest-F1 model (client {best_cid}, "
            f"client_id={best_client_id}, accuracy={best_metrics['accuracy']:.4f}, "
            f"f1={best_metrics['f1']:.4f}) and saved it to {model_path}"
        )
        aggregated_parameters = Parameters(
            tensors=[best_payload], tensor_type="xgboost-ubj"
        )
        return aggregated_parameters, {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
        }


class XGBoostBaggingStrategy(XGBoostStrategy):
    def __init__(
        self,
        model_dir: str,
        num_clients: int,
        val_data_path: Optional[str] = None,
        category_maps_path: str = "category_maps.json",
        leaf_scale: float = 1.0,
    ):
        super().__init__(model_dir, num_clients, val_data_path, category_maps_path)
        self.global_model_json: Optional[bytes] = None
        self.leaf_scale = leaf_scale

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # scale and bagging

        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(
                f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}"
            )

        print(
            f"[Info] Bagging Round {server_round}: {len(results)}/{self.num_clients} "
        )

        merged_json = self.global_model_json
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if not payload:
                continue

            reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
            bst = xgb.Booster()
            bst.load_model(bytearray(payload))
            payload_json = bst.save_raw("json")
            num_trees_received = len(
                json.loads(bytearray(payload_json))["learner"]["gradient_booster"][
                    "model"
                ]["trees"]
            )
            print(
                f"[Bagging] Client {client_proxy.cid} (client_id={reported_client_id}, "
                f"sent {num_trees_received} tree(s) this round."
            )

            payload_json = scale_leaf_values(payload_json, self.leaf_scale)
            solo_json = aggregate_bagging_verified(self.global_model_json, payload_json)
            solo_bst = xgb.Booster()
            solo_bst.load_model(bytearray(solo_json))
            solo_metrics = self._evaluate_model_on_server(
                bytes(solo_bst.save_raw("ubj"))
            )
            print(
                f"[Bagging][ClientScore] Round {server_round} client_id={reported_client_id} "
                f"(cid={client_proxy.cid}) on validation (chunk_6): "
                f"f1={solo_metrics['f1']:.4f} accuracy={solo_metrics['accuracy']:.4f} "
                f"precision={solo_metrics['precision']:.4f} recall={solo_metrics['recall']:.4f}"
            )
            merged_json = aggregate_bagging_verified(merged_json, payload_json)

        self.global_model_json = merged_json
        bst_merged = xgb.Booster()
        bst_merged.load_model(bytearray(merged_json))
        merged_ubj = bytes(bst_merged.save_raw("ubj"))
        merged_metrics = self._evaluate_model_on_server(merged_ubj)
        model_path = os.path.join(
            self.model_dir, f"global_model_round_{server_round}.ubj"
        )

        with open(model_path, "wb") as f:
            f.write(merged_ubj)
        with open(self.latest_model_path, "wb") as f:
            f.write(merged_ubj)

        print(
            f"[Info] Round {server_round} bagging-merged {len(results)} client models "
            f"(accuracy={merged_metrics['accuracy']:.4f}, f1={merged_metrics['f1']:.4f}, "
            f"margin=[{merged_metrics['margin_min']:.2f}, {merged_metrics['margin_max']:.2f}], "
            f"margin_mean={merged_metrics['margin_mean']:.2f}) "
            f"and saved it to {model_path}"
        )

        aggregated_parameters = Parameters(
            tensors=[merged_ubj], tensor_type="xgboost-ubj"
        )

        return aggregated_parameters, {
            "accuracy": merged_metrics["accuracy"],
            "precision": merged_metrics["precision"],
            "recall": merged_metrics["recall"],
            "f1": merged_metrics["f1"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Flower server with XGBoost strategy."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=1, help="Number of rounds to train the model."
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Number of clients to train the model.",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Flower server address",
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default="val-data/val.csv",
        help="Path to the validation data CSV file.",
    )
    parser.add_argument(
        "--category_maps_path",
        type=str,
        default="category_maps.json",
        help="Path to the shared category_maps.json produced once from the full "
        "dataset (must match the file every client.py uses).",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="winner",
        choices=["winner", "bagging"],
        help="winner = max() picks the single highest-F1 client model each round (default, existing behavior). "
        "bagging = Flower's official bagging aggregate() concatenates every client's trees, none discarded. ",
    )

    parser.add_argument(
        "--leaf_scale",
        type=float,
        default=0.5,
        help="bagging/err_lfr aggregation mode: Adjusting the scaling when using tree bagging",
    )
    args = parser.parse_args()

    if args.aggregation == "bagging":
        strategy = XGBoostBaggingStrategy(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            category_maps_path=args.category_maps_path,
            leaf_scale=args.leaf_scale,
        )
    else:
        strategy = XGBoostStrategy(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            category_maps_path=args.category_maps_path,
        )

    print("[Info] Flower Server (XGBoost) is starting...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )