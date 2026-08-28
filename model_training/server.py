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
import json

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


def scale_leaf_values(model_json: bytes, w: float) -> bytes:
    """Scale every leaf node's value in a serialized booster's trees by w.

    Fixes the margin-explosion problem in notes/13a-bagging-baseline.md: 5
    clients each independently boost 10 new trees against the SAME shared
    global model every round, and aggregate_bagging_verified() sums (not
    averages) all 5 contributions -- equivalent to a 5x learning-rate
    inflation, compounding round over round until margins saturate the
    sigmoid (round 1: [-8, 7]; round 10: [-14214, 2283], see notes/13a).

    Insertion point per notes/00-findings.md finding 3/notes/02 section 4:
    since XGBoost's prediction is base_score + sum of every tree's selected
    leaf value, scaling all leaf values in a client's new-tree batch by w
    BEFORE it is merged into the global model scales that batch's
    contribution to the ensemble by exactly w -- verified empirically
    (leaf values live in BOTH base_weights[i] and split_conditions[i] at
    leaf nodes, i.e. where left_children[i] == -1; scaling both by w=0.3
    on a real client_1_round_1.ubj model reproduced a margin-contribution
    ratio of 0.29999995-0.30000004 against the unscaled model, max error
    9e-8 -- this resolves notes/02's "[推測，待驗證]" on the leaf formula).
    Both fields are scaled together (not just one) so a client continuing
    to boost on top of this model later (xgb_model=...) sees a consistent
    tree, not a base_weights/split_conditions mismatch.
    """
    model = json.loads(bytearray(model_json))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    for tree in trees:
        left_children = tree["left_children"]
        base_weights = tree["base_weights"]
        split_conditions = tree["split_conditions"]
        for i, left in enumerate(left_children):
            if left == -1:
                base_weights[i] = base_weights[i] * w
                split_conditions[i] = split_conditions[i] * w
    return bytes(json.dumps(model), "utf-8")


def aggregate_bagging_verified(bst_prev_org: Optional[bytes], bst_curr_org: bytes) -> bytes:
    """Bagging aggregation: append the trees in bst_curr_org to bst_prev_org.

    This is derived from Flower's official aggregate()
    (flwr/server/strategy/fedxgb_bagging.py:118-154) but fixes one thing:
    the official version decides how many trees to append by reading
    gbtree_model_param.num_parallel_tree from bst_curr_org -- a fixed
    XGBoost hyperparameter (how many trees to train in parallel per
    boosting iteration, normally 1), NOT "how many new trees this
    submission contains". It only happens to give the right answer when a
    client trains exactly num_parallel_tree (=1) new tree per round, which
    is what Flower's own bagging example does (local-epochs=1). Our
    client.py trains NUM_BOOST_ROUND=10 trees per round, so that field is
    always 1 regardless of the true delta size -- using it silently
    merged the same stale first tree every round instead of each round's
    actual new trees. See notes/00-findings.md finding 21 and
    notes/13a-bagging-baseline.md for how this was found.

    Fix: don't read any declared/self-reported count. client.py now sends
    ONLY this round's new trees (see _parameters_from_new_trees in
    client.py), so the number to append is simply
    len(trees in bst_curr_org) -- counted directly from the payload
    actually received, not trusted from any field a client could
    misreport. This is the same "don't trust a node's own claim, verify
    the payload itself" principle as the fallback removed in ff82d04.
    """
    if not bst_prev_org:
        return bst_curr_org

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    gbtree_prev = bst_prev["learner"]["gradient_booster"]["model"]
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    tree_num_prev = int(gbtree_prev["gbtree_model_param"]["num_trees"])
    num_new_trees = len(trees_curr)

    gbtree_prev["gbtree_model_param"]["num_trees"] = str(tree_num_prev + num_new_trees)
    iteration_indptr = gbtree_prev["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + num_new_trees)

    for tree_count in range(num_new_trees):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        gbtree_prev["trees"].append(trees_curr[tree_count])
        gbtree_prev["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


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
        zero_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "margin_min": 0.0, "margin_max": 0.0, "margin_mean": 0.0,
        }
        if self.dval is None:
            return zero_metrics
        try:
            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            preds = bst.predict(self.dval)
            margin = bst.predict(self.dval, output_margin=True)

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
                "margin_min": float(np.min(margin)),
                "margin_max": float(np.max(margin)),
                "margin_mean": float(np.mean(margin)),
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


class XGBoostBaggingStrategy(XGBoostStrategy):
    """Bagging aggregation baseline for notes/13a-bagging-baseline.md.

    aggregate_fit() concatenates every client's new trees into the running
    global model each round -- no client is ever discarded, unlike
    XGBoostStrategy's max() which keeps only one winner.

    Two things had to be fixed relative to a first attempt that just used
    Flower's official aggregate() directly (see notes/00-findings.md
    finding 21, notes/13a-bagging-baseline.md for the full investigation):

    1. Merge logic: Flower's official aggregate() decides how many trees
       to append by reading gbtree_model_param.num_parallel_tree -- a fixed
       XGBoost hyperparameter (normally 1), not "how many new trees this
       submission has". With this project's NUM_BOOST_ROUND=10, that field
       is always 1 regardless of true delta size, so it silently merged
       the same stale first tree every round. aggregate_bagging_verified()
       (above) fixes this by counting len(trees) in the actual payload
       instead -- ground truth from the payload's own structure, never
       trusted from any client-declared value or hyperparameter that
       doesn't reflect it.
    2. Wire format: client.py now sends only this round's new trees (see
       _parameters_from_new_trees), not the whole cumulative model --
       required for (1) to have a well-defined "how many new trees" to
       count in the first place.

    Flower's aggregate() parses model bytes with json.loads(), so bagging
    needs JSON, not the UBJ bytes this project's tensor_type="xgboost-ubj"
    actually transmits. Verified empirically that loading UBJ bytes into a
    Booster and re-serializing via save_raw("json") converts correctly,
    and converting the merged JSON result back via save_raw("ubj")
    round-trips losslessly (predictions identical) -- confirmed with a
    synthetic model before relying on it here. The round-trip back to UBJ
    is necessary because client.py's _load_model_from_bytes() loads by
    file path with a ".ubj" suffix, and xgboost's load_model(path) trusts
    the extension -- JSON content in a ".ubj"-named file fails to load
    (tested directly: json.cc parse error), so the wire format must stay
    real UBJ.
    """

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 leaf_scale: float = 1.0):
        super().__init__(model_dir, num_clients, val_data_path)
        self.global_model_json: Optional[bytes] = None
        # See scale_leaf_values() docstring and notes/13a-bagging-baseline.md
        # "未解問題" 1 -- fixes the margin-explosion problem from summing
        # (not averaging) 5 clients' independent corrections each round.
        # Configurable, NOT hardcoded to 1/num_clients: that's one
        # hypothesis (matches NVFlare's lr_mode="uniform" default, see
        # notes/03/notes/00-findings.md finding 5), not an established fact
        # for this codebase's num_boost_round=10-per-round setup.
        self.leaf_scale = leaf_scale

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

        # Each client now sends ONLY this round's new trees (see
        # client.py's _parameters_from_new_trees), so the payload here is
        # NOT a full model -- evaluating it against chunk_6 in isolation
        # would not mean "this client's model quality" (a lone tree slice's
        # raw output isn't a meaningful probability without the rest of the
        # ensemble it was boosted on top of), so that per-client eval step
        # from the winner-take-all strategy is intentionally dropped here.
        # What's logged instead is the tree count actually found in each
        # payload -- counted from the payload itself, never from anything
        # a client declares -- so a mismatch (e.g. a client sending more or
        # fewer trees than expected) is visible in the log.
        merged_json = self.global_model_json
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if not payload:
                continue
            reported_client_id = (fit_res.metrics or {}).get("client_id", "?")

            bst = xgb.Booster()
            bst.load_model(bytearray(payload))
            payload_json = bst.save_raw("json")
            num_trees_received = len(json.loads(bytearray(payload_json))
                                      ["learner"]["gradient_booster"]["model"]["trees"])
            print(
                f"[Bagging] Client {client_proxy.cid} (client_id={reported_client_id}) "
                f"sent {num_trees_received} tree(s) this round."
            )

            # Scale THIS client's new-tree batch before merging -- server-side,
            # not client-side, so a malicious client can't opt out of the
            # scaling by simply not applying it (zero-trust, same rationale
            # as len(trees_curr) below not trusting any client-declared
            # count -- see aggregate_bagging_verified()'s docstring).
            payload_json = scale_leaf_values(payload_json, self.leaf_scale)

            merged_json = aggregate_bagging_verified(merged_json, payload_json)

        self.global_model_json = merged_json

        # Convert the merged model back to real UBJ bytes for wire
        # transmission -- see class docstring for why this round-trip
        # is required rather than optional.
        bst_merged = xgb.Booster()
        bst_merged.load_model(bytearray(merged_json))
        merged_ubj = bytes(bst_merged.save_raw("ubj"))

        merged_metrics = self._evaluate_model_on_server(merged_ubj)

        model_path = os.path.join(self.model_dir, f"global_model_round_{server_round}.ubj")
        with open(model_path, "wb") as f: f.write(merged_ubj)
        with open(self.latest_model_path, "wb") as f: f.write(merged_ubj)

        print(f"[Info] Round {server_round} bagging-merged {len(results)} client models "
              f"(accuracy={merged_metrics['accuracy']:.4f}, f1={merged_metrics['f1']:.4f}, "
              f"margin=[{merged_metrics['margin_min']:.2f}, {merged_metrics['margin_max']:.2f}], "
              f"margin_mean={merged_metrics['margin_mean']:.2f}) "
              f"and saved it to {model_path}")

        aggregated_parameters = Parameters(tensors=[merged_ubj], tensor_type="xgboost-ubj")
        return aggregated_parameters, {
            "accuracy": merged_metrics["accuracy"],
            "precision": merged_metrics["precision"],
            "recall": merged_metrics["recall"],
            "f1": merged_metrics["f1"],
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
    parser.add_argument("--aggregation", type=str, default="winner", choices=["winner", "bagging"],
                        help="winner = max() picks the single highest-F1 client model each round (default, existing behavior). "
                             "bagging = Flower's official bagging aggregate() concatenates every client's trees, none discarded.")
    parser.add_argument("--leaf_scale", type=float, default=1.0,
                        help="bagging mode only: multiply each client's new-tree leaf values by this "
                             "before merging (see scale_leaf_values()). 1.0 = no scaling (original bug). "
                             "Fixes the margin-explosion problem from summing 5 independent corrections "
                             "each round -- see notes/13a-bagging-baseline.md.")
    args = parser.parse_args()

    strategy_cls = XGBoostBaggingStrategy if args.aggregation == "bagging" else XGBoostStrategy
    if args.aggregation == "bagging":
        strategy = strategy_cls(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            leaf_scale=args.leaf_scale,
        )
    else:
        strategy = strategy_cls(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
        )

    print("[Info] Flower Server (XGBoost) is starting...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )