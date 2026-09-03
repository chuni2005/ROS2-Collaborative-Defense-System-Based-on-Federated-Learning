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

# precision/recall/F1 的正類是 1（「攻擊」）——ROSPaCe 的攻擊樣本是多數類
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
    """把一批樹的所有葉節點值乘上 w，讓這批樹對預測結果的影響縮小成 w 倍。
        用來壓低多節點加總造成的預測值膨脹（5 個節點各自修正、伺服器加總不取平均，等於學習率變 5 倍）。   
    """
    model = json.loads(bytearray(model_json))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    for tree in trees:
        left_children = tree["left_children"]
        base_weights = tree["base_weights"]
        split_conditions = tree["split_conditions"]
        for i, left in enumerate(left_children):
            if left == -1:#是否為葉節點
                base_weights[i] = base_weights[i] * w
                split_conditions[i] = split_conditions[i] * w
    return bytes(json.dumps(model), "utf-8")


def aggregate_bagging_verified(bst_prev_org: Optional[bytes], bst_curr_org: bytes) -> bytes:
    #合併兩批樹成一個模型，改自 Flower aggregate()，只信任實際樹數。
    if not bst_prev_org:
        return bst_curr_org

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    gbtree_prev = bst_prev["learner"]["gradient_booster"]["model"]
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    tree_num_prev = int(gbtree_prev["gbtree_model_param"]["num_trees"])
    num_new_trees = len(trees_curr)  # ← 改自 Flower：數實際樹數

    gbtree_prev["gbtree_model_param"]["num_trees"] = str(tree_num_prev + num_new_trees)
    iteration_indptr = gbtree_prev["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + num_new_trees)  # 疊代記法不一致
    # Flower 原始程式碼（fedxgb_bagging.py:139-141）：
    #     bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
    #         iteration_indptr[-1] + paral_tree_num_curr
    #     )
    # 改因：num_parallel_tree 固定是 1，我們每輪實際練 10 棵

    for tree_count in range(num_new_trees):  # ← 改自 Flower：range 用實際樹數
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count  # 轉成全域序號
        gbtree_prev["trees"].append(trees_curr[tree_count])
        gbtree_prev["tree_info"].append(0)  # 固定寫 0，未逐值驗證

    return bytes(json.dumps(bst_prev), "utf-8")


def merge_payloads(base_json: Optional[bytes], payload_jsons: List[bytes],
                    exclude_indices: Optional[set] = None) -> Optional[bytes]:
    """依序把 payload_jsons 合併到 base_json 上；exclude_indices 是要跳過的
    位置索引（不是 client_id），預設空集合即合併全部，供 ERR/LFR 留一法複用。
    """
    exclude_indices = exclude_indices or set()
    merged = base_json
    for i, payload_json in enumerate(payload_jsons):
        if i in exclude_indices:
            continue
        merged = aggregate_bagging_verified(merged, payload_json)
    return merged


class XGBoostStrategy(fl.server.strategy.FedAvg):
    # 贏者全拿：每輪保留 F1 最高的單一模型

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 test_natural_data_path: Optional[str] = None, test_rare_data_path: Optional[str] = None):
        """載入伺服器驗證資料，找不到就拋錯中止，不信任 client 回報的分數。
        test_natural_data_path / test_rare_data_path 是選用的，只給訓練
        結束後的最終報告用；不給路徑就不載入，_evaluate_model_on_server()
        對應的資料來源就評不出東西（回傳全 0）。
        """
        self.num_clients = num_clients
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.latest_model_path = os.path.join(self.model_dir, "global_model_latest.ubj")

        self.dval, self.y_true = self._load_eval_dataset(val_data_path, "validation", required=True)
        self.dtest_natural, self.y_test_natural = self._load_eval_dataset(test_natural_data_path, "test_natural")
        self.dtest_rare, self.y_test_rare = self._load_eval_dataset(test_rare_data_path, "test_rare")

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
        )

    def _load_eval_dataset(self, data_path: Optional[str], label: str, required: bool = False):
        """讀一份 CSV、preprocess 後轉成 (DMatrix, 標籤陣列)，只在 __init__ 呼叫
        一次供之後重複查詢，不會每次評估都重讀 CSV。required=True（validation
        用）時路徑缺失直接中止伺服器啟動，不退回信任 client 自報分數；
        required=False（test_natural/test_rare 用，僅供訓練結束後的最終報告
        輸出，不參與訓練期間的模型選擇）時路徑沒給就回傳 (None, None)。
        """
        if not data_path:
            if required:
                raise FileNotFoundError(
                    f"Server-side {label} data not found at {data_path!r}. "
                    "Refusing to start: this system does not fall back to trusting "
                    "client-reported metrics for model selection."
                )
            return None, None
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Server-side {label} data path given but not found: {data_path!r}")

        print(f"[Info] Loading server-side {label} data from {data_path}...")
        df = pd.read_csv(data_path)
        df = preprocess_data(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return xgb.DMatrix(X, label=y), y.values

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

    # zero_division=0：某類別完全缺席時回傳 0.0，不拋例外
    def _evaluate_model_on_server(self, model_bytes: bytes, dataset: str = "validation") -> Dict[str, float]:
        """用 dataset 對應的快取 DMatrix 評估模型，回傳 accuracy/precision/
        recall/f1/margin/logloss。f1 供贏者全拿模式排序選模型，logloss 供
        ERR/LFR 的 LFR 計算剔除分數，margin 只用於 log 診斷（追蹤 leaf-scale
        疊加是否暴衝）。aggregate_fit() 只能用預設的 "validation"；
        "test_natural"/"test_rare" 僅供訓練後最終報告用，混進模型選擇會讓
        報告數字失去參考價值。失敗或資料未載入回傳 zero_metrics
        （logloss=1.0 代表最差情況，非 0.0）。
        """
        zero_metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "margin_min": 0.0, "margin_max": 0.0, "margin_mean": 0.0,
            "logloss": 1.0,
        }
        datasets = {
            "validation": (self.dval, self.y_true),
            "test_natural": (self.dtest_natural, self.y_test_natural),
            "test_rare": (self.dtest_rare, self.y_test_rare),
        }
        dmatrix, y_true = datasets.get(dataset, (None, None))
        if dmatrix is None:
            return zero_metrics
        try:
            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            preds = bst.predict(dmatrix)
            margin = bst.predict(dmatrix, output_margin=True)

            preds_binary = [1 if p > 0.5 else 0 for p in preds]
            correct = sum(1 for p, y in zip(preds_binary, y_true) if p == y)
            accuracy = correct / len(y_true)
            precision = precision_score(y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)
            recall = recall_score(y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)
            f1 = f1_score(y_true, preds_binary, pos_label=POSITIVE_CLASS, zero_division=0)

            eps = 1e-7
            preds_clipped = np.clip(preds, eps, 1 - eps)
            y_true_arr = np.asarray(y_true, dtype=np.float64)
            logloss = float(-np.mean(
                y_true_arr * np.log(preds_clipped) + (1 - y_true_arr) * np.log(1 - preds_clipped)
            ))

            return {
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "margin_min": float(np.min(margin)),
                "margin_max": float(np.max(margin)),
                "margin_mean": float(np.mean(margin)),
                "logloss": logloss,
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
        """每輪評分所有節點模型，只留 F1 最高的一個當全域模型。"""

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
                # 只用來記錄身分，不用於評分／信任判斷（可被謊報）
                reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
                print(
                    f"[Server Eval] Client {client_proxy.cid} (client_id={reported_client_id}) "
                    f"Accuracy: {server_metrics['accuracy']:.4f} "
                    f"Precision: {server_metrics['precision']:.4f} "
                    f"Recall: {server_metrics['recall']:.4f} "
                    f"F1: {server_metrics['f1']:.4f}"
                )
                payloads.append((payload, server_metrics, client_proxy.cid, reported_client_id))

        # 用 F1 排序：跟「永遠猜攻擊」地板分差約 19.6%，足夠分高下
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
    """每輪把所有節點新樹接進全域模型，取代贏者全拿。
    merge 用 JSON，傳輸用 UBJ，兩者間需轉換（已驗證 UBJ→JSON→UBJ 不失真）。
    """

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 test_natural_data_path: Optional[str] = None, test_rare_data_path: Optional[str] = None,
                 leaf_scale: float = 1.0):
        """多存跨輪累積的全域模型（JSON）與葉值縮放係數 leaf_scale。"""
        super().__init__(model_dir, num_clients, val_data_path, test_natural_data_path, test_rare_data_path)
        self.global_model_json: Optional[bytes] = None
        # 可設定，不寫死 1/節點數：測過候選值後那不是最好的
        self.leaf_scale = leaf_scale

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """縮放並合併這一輪所有節點的新樹，變成新的全域模型；每個節點另外
        算一個診斷用的單獨分數（見迴圈內 solo_metrics），只記錄不篩選——
        真正決定全域模型的還是下面 merged_json 依序合併的結果。
        """
        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}")

        print(
            f"[Bagging] Round {server_round}: {len(results)}/{self.num_clients} "
            f"client(s) participated in this round's aggregation."
        )

        merged_json = self.global_model_json
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if not payload:
                continue
            reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
            # 純觀察用途，跟 client_id 一樣是 client 自報、伺服器沒有獨立驗證
            # 管道——只用來讓下面這行 log 好讀，不影響縮放、合併、或任何判斷。
            reported_malicious = (fit_res.metrics or {}).get("malicious", False)

            bst = xgb.Booster()
            bst.load_model(bytearray(payload))
            payload_json = bst.save_raw("json")
            num_trees_received = len(json.loads(bytearray(payload_json))
                                      ["learner"]["gradient_booster"]["model"]["trees"])
            print(
                f"[Bagging] Client {client_proxy.cid} (client_id={reported_client_id}, "
                f"malicious={reported_malicious}) sent {num_trees_received} tree(s) this round."
            )

            # 在伺服器端縮放，節點無法規避（零信任）
            payload_json = scale_leaf_values(payload_json, self.leaf_scale)

            # 這個節點單獨接上「上一輪存檔的全域模型」（不是這一輪目前為止
            # 已經合併好的 merged_json）算出來的候選分數，只用來記錄、方便
            # 事後比對哪個節點這一輪的分數特別低，不影響下面 merged_json 的
            # 實際合併順序與結果，也不用於任何篩選或排除判斷——這個類別本身
            # 沒有防禦，真的要依分數篩選節點見 XGBoostErrLfrStrategy。
            solo_json = aggregate_bagging_verified(self.global_model_json, payload_json)
            solo_bst = xgb.Booster()
            solo_bst.load_model(bytearray(solo_json))
            solo_metrics = self._evaluate_model_on_server(bytes(solo_bst.save_raw("ubj")))
            print(
                f"[Bagging][ClientScore] Round {server_round} client_id={reported_client_id} "
                f"(cid={client_proxy.cid}, malicious={reported_malicious}) on validation (chunk_6): "
                f"f1={solo_metrics['f1']:.4f} accuracy={solo_metrics['accuracy']:.4f} "
                f"precision={solo_metrics['precision']:.4f} recall={solo_metrics['recall']:.4f}"
            )

            merged_json = aggregate_bagging_verified(merged_json, payload_json)

        self.global_model_json = merged_json

        # 轉回 UBJ 才能傳輸：節點按副檔名判斷格式，JSON 內容會載入失敗
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


class XGBoostErrLfrStrategy(XGBoostBaggingStrategy):
    """bagging 之前多一層 ERR/LFR 過濾（Fang et al., USENIX Security 2020）：
    對每個節點 i，比較候選模型 A（納入全部節點）與 B_i（排除節點 i）在驗證
    集上的表現，差異越大代表拿掉它模型變得越好、該節點越可疑。ERR 用
    error rate（1-accuracy），LFR 用 cross-entropy loss，各自取影響最大的
    num_to_exclude 個節點，剔除名單是兩者聯集。
    """

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 test_natural_data_path: Optional[str] = None, test_rare_data_path: Optional[str] = None,
                 leaf_scale: float = 1.0, num_to_exclude: int = 1):
        """num_to_exclude 是 ERR、LFR 各自要剔除的節點數（對應 Fang et al.
        論文的 c，即攻擊者數量的估計上限）。預設值 1 配合本次實驗設定
        （5 節點中 1 個惡意節點），不是通用建議值——真實場景下攻擊者數量
        未知時如何設定，本實作未解決。
        """
        super().__init__(model_dir, num_clients, val_data_path, test_natural_data_path,
                          test_rare_data_path, leaf_scale)
        self.num_to_exclude = num_to_exclude

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """ERR/LFR 留一法剔除可疑節點後，用 XGBoostBaggingStrategy 同一套
        aggregate_bagging_verified()（經 merge_payloads() 包一層）合併存活節點。
        """
        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}")

        print(
            f"[ErrLfr] Round {server_round}: {len(results)}/{self.num_clients} "
            f"client(s) participated in this round's aggregation."
        )

        # --- 收集這一輪每個節點縮放後的 payload，位置索引固定下來供後面留一法引用 ---
        client_infos = []  # 每個元素: (client_proxy, reported_client_id, reported_malicious, payload_json)
        for client_proxy, fit_res in results:
            payload = self._extract_payload(fit_res.parameters)
            if not payload:
                continue
            reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
            reported_malicious = (fit_res.metrics or {}).get("malicious", False)

            bst = xgb.Booster()
            bst.load_model(bytearray(payload))
            payload_json = bst.save_raw("json")
            payload_json = scale_leaf_values(payload_json, self.leaf_scale)
            client_infos.append((client_proxy, reported_client_id, reported_malicious, payload_json))

        n = len(client_infos)
        payload_jsons = [info[3] for info in client_infos]

        # --- 候選模型 A：這一輪納入全部節點 ---
        model_a_json = merge_payloads(self.global_model_json, payload_jsons)
        bst_a = xgb.Booster()
        bst_a.load_model(bytearray(model_a_json))
        metrics_a = self._evaluate_model_on_server(bytes(bst_a.save_raw("ubj")))
        print(
            f"[ErrLfr] Round {server_round} candidate A (all {n} clients) on validation "
            f"(chunk_6): accuracy={metrics_a['accuracy']:.4f} logloss={metrics_a['logloss']:.4f}"
        )

        # --- 對每個節點 i，候選模型 B_i：排除節點 i 之後合併其餘節點 ---
        err_impact = {}
        lfr_impact = {}
        for i, (client_proxy, reported_client_id, reported_malicious, _) in enumerate(client_infos):
            model_bi_json = merge_payloads(self.global_model_json, payload_jsons, exclude_indices={i})
            if model_bi_json is None:
                # 只有 1 個節點參與、又被排除自己時才會發生：沒有東西可合併，
                # 視為「拿掉它模型無限差」，強制排最前面優先剔除。
                err_impact[i] = float("inf")
                lfr_impact[i] = float("inf")
                print(f"[ErrLfr] Round {server_round} client_id={reported_client_id}: "
                      f"only participant this round, B_i is empty, forcing max impact.")
                continue
            bst_bi = xgb.Booster()
            bst_bi.load_model(bytearray(model_bi_json))
            metrics_bi = self._evaluate_model_on_server(bytes(bst_bi.save_raw("ubj")))

            err_i = (1.0 - metrics_a["accuracy"]) - (1.0 - metrics_bi["accuracy"])  # E_A - E_Bi
            lfr_i = metrics_a["logloss"] - metrics_bi["logloss"]  # L_A - L_Bi
            err_impact[i] = err_i
            lfr_impact[i] = lfr_i
            print(
                f"[ErrLfr][ClientImpact] Round {server_round} client_id={reported_client_id} "
                f"(cid={client_proxy.cid}, malicious={reported_malicious}) B_i accuracy="
                f"{metrics_bi['accuracy']:.4f} logloss={metrics_bi['logloss']:.4f} "
                f"err_impact={err_i:.6f} lfr_impact={lfr_i:.6f}"
            )

        # --- 剔除規則：ERR、LFR 各自取影響最大的 num_to_exclude 個，取聯集 ---
        k = min(self.num_to_exclude, n)
        err_ranked = sorted(range(n), key=lambda i: err_impact[i], reverse=True)
        lfr_ranked = sorted(range(n), key=lambda i: lfr_impact[i], reverse=True)
        err_excluded = set(err_ranked[:k])
        lfr_excluded = set(lfr_ranked[:k])
        excluded = err_excluded | lfr_excluded

        excluded_client_ids = [client_infos[i][1] for i in sorted(excluded)]
        print(
            f"[ErrLfr] Round {server_round} excluded client_id(s)={excluded_client_ids} "
            f"(ERR flagged idx={sorted(err_excluded)}, LFR flagged idx={sorted(lfr_excluded)}, "
            f"union size={len(excluded)}/{n})"
        )

        # --- 正式合併：只納入沒被剔除的節點 ---
        merged_json = merge_payloads(self.global_model_json, payload_jsons, exclude_indices=excluded)
        if merged_json is None:
            print(f"[Warning] Round {server_round}: all clients excluded, nothing to merge; "
                  f"keeping previous global model unchanged.")
            merged_json = self.global_model_json
        self.global_model_json = merged_json

        # 轉回 UBJ 才能傳輸：節點按副檔名判斷格式，JSON 內容會載入失敗
        bst_merged = xgb.Booster()
        bst_merged.load_model(bytearray(merged_json))
        merged_ubj = bytes(bst_merged.save_raw("ubj"))

        merged_metrics = self._evaluate_model_on_server(merged_ubj)

        model_path = os.path.join(self.model_dir, f"global_model_round_{server_round}.ubj")
        with open(model_path, "wb") as f: f.write(merged_ubj)
        with open(self.latest_model_path, "wb") as f: f.write(merged_ubj)

        print(f"[Info] Round {server_round} ERR/LFR-filtered bagging kept {n - len(excluded)}/{n} "
              f"client models (excluded={excluded_client_ids}) "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Flower server with XGBoost strategy.")
    parser.add_argument("--model_dir", type=str, default="./outputs/models", help="Directory to save the trained model.")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds to train the model.")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients to train the model.")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080", help="Flower server address")
    parser.add_argument("--validation_data_path", type=str, default="split_data/chunk_6.csv", help="Path to the validation data CSV file.")
    parser.add_argument("--test_natural_data_path", type=str, default=None,
                        help="Optional CSV for final overall-F1 reporting only. Unset by default so "
                             "training runs never load it; aggregate_fit() never reads it either way.")
    parser.add_argument("--test_rare_data_path", type=str, default=None,
                        help="Optional CSV for final per-attack-type recall reporting only. Unset by "
                             "default so training runs never load it; aggregate_fit() never reads it either way.")
    parser.add_argument("--aggregation", type=str, default="winner", choices=["winner", "bagging", "err_lfr"],
                        help="winner = max() picks the single highest-F1 client model each round (default, existing behavior). "
                             "bagging = Flower's official bagging aggregate() concatenates every client's trees, none discarded. "
                             "err_lfr = bagging with an ERR/LFR leave-one-out filter applied first (see XGBoostErrLfrStrategy).")
    parser.add_argument("--leaf_scale", type=float, default=1.0,
                        help="bagging/err_lfr mode only: multiply each client's new-tree leaf values by this "
                             "before merging (see scale_leaf_values()). 1.0 = no scaling (original bug). "
                             "Fixes the margin-explosion problem from summing 5 independent corrections "
                             "each round -- see notes/13a-bagging-baseline.md.")
    parser.add_argument("--num_to_exclude", type=int, default=1,
                        help="err_lfr mode only: how many clients ERR and how many clients LFR each flag "
                             "as suspicious per round (the union of both is excluded from this round's "
                             "merge). See XGBoostErrLfrStrategy's docstring for why the default is 1.")
    args = parser.parse_args()

    if args.aggregation == "bagging":
        strategy = XGBoostBaggingStrategy(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            test_natural_data_path=args.test_natural_data_path,
            test_rare_data_path=args.test_rare_data_path,
            leaf_scale=args.leaf_scale,
        )
    elif args.aggregation == "err_lfr":
        strategy = XGBoostErrLfrStrategy(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            test_natural_data_path=args.test_natural_data_path,
            test_rare_data_path=args.test_rare_data_path,
            leaf_scale=args.leaf_scale,
            num_to_exclude=args.num_to_exclude,
        )
    else:
        strategy = XGBoostStrategy(
            model_dir=args.model_dir,
            num_clients=args.num_clients,
            val_data_path=args.validation_data_path,
            test_natural_data_path=args.test_natural_data_path,
            test_rare_data_path=args.test_rare_data_path,
        )

    print("[Info] Flower Server (XGBoost) is starting...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )