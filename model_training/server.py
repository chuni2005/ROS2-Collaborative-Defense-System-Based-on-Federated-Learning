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
# （約佔 78%），跟一般 IDS 資料集直覺相反。這裡的 precision 是「模型判定為
# 攻擊的裡面，有多少真的是攻擊」；recall 是「真正的攻擊裡面，抓到了多少」。
# 見 notes/11-dataset-check.md 與 notes/00-findings.md 發現 17。
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

    原理：XGBoost 的預測值 = base_score + 每棵樹走到的葉節點值加總。
    所以把葉節點值乘 0.5，這批樹的影響力就剛好變成一半。

    輸入：model_json — 一個節點這輪新練的 10 棵樹，序列化成 JSON bytes
          w          — 縮放係數，目前用 0.5
    輸出：結構完全一樣的 JSON bytes，只有葉節點的數值變了

    怎麼認出葉節點：XGBoost 把一棵樹存成幾個平行的陣列。
    left_children[i] == -1 代表第 i 個節點沒有左子樹，也就是葉節點。
    葉節點的值同時存在 base_weights[i] 和 split_conditions[i] 兩個欄位
    （XGBoost 的內部設計，同一個數字放兩個地方），兩個都要改——
    只改一個的話，節點之後拿這個模型繼續訓練時會讀到不一致的值。

    為什麼需要縮放：5 個節點每輪各自對同一個全域模型練 10 棵樹，
    伺服器把 5 份都接起來，等於學習率變成 5 倍。幾輪之後預測值會爆掉——
    實測不縮放時第 10 輪的 margin 範圍是 [-14214, 2283]，
    正常應該在 [-10, 10] 附近。

    完整調查過程見 notes/13a-leaf-scale-fix.md。
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
    """把兩批樹接在一起，變成一個更大的模型，同時更新幾個記帳用的欄位。

    原理：一個 XGBoost 模型的預測值是 base_score 加上模型裡每一棵樹的貢獻
    加總，所有的樹放在同一個陣列裡，樹跟樹之間互不影響、彼此獨立。所以
    「合併兩個模型」在結構上就是「把兩個樹陣列接起來」，不需要重新訓練，
    也不會改變任何一棵樹本身的內容。

    輸入：bst_prev_org — 這一輪目前為止已經合併好的全域模型，JSON bytes；
          這一輪第一個被處理的節點時是 None（還沒有任何樹被接進來）
          bst_curr_org — 剛收到的一個節點這一輪新練的樹，JSON bytes
          （已經被 scale_leaf_values() 縮放過）
    輸出：合併後的模型，JSON bytes——bst_curr_org 的樹全部接到
          bst_prev_org 的樹陣列後面

    怎麼做：一個 XGBoost 模型的 JSON 裡，跟樹有關的幾個欄位都在
    learner.gradient_booster.model 底下：

    - trees：所有樹的陣列，每一棵樹是一個字典（分裂條件、左右子節點、
      葉值等）
    - gbtree_model_param.num_trees：字串形式記下目前模型總共有幾棵樹
    - iteration_indptr：一個遞增的索引陣列，記錄「第幾次訓練疊代」對應到
      trees 陣列的哪個區間（正常訓練時，一次疊代通常對應一棵新樹）
    - tree_info：每棵樹屬於哪個輸出群組的編號（只有多分類這種多輸出模型
      才會用到多個群組；這個專案是二元分類，永遠只有一組，固定是 0）

    合併步驟：如果 bst_prev_org 是 None，代表沒有東西可以接，直接把
    bst_curr_org 當這一輪的起點回傳。否則，數出 bst_curr_org 裡有幾棵新樹
    （num_new_trees），把 num_trees 加上這個數字，在 iteration_indptr 後面
    補一筆（代表這批新樹佔用的疊代區間），然後逐棵把新樹接到 bst_prev_org
    的 trees 陣列後面——同時把每棵樹的 id 改成接在 bst_prev_org 現有樹後面
    的全域序號（避免跟已經存在的樹編號重複），並在 tree_info 補一筆 0。

    為什麼需要它：這是整個 bagging 聚合的地基——沒有這個函式，就沒有辦法讓
    多個節點的訓練成果同時保留在同一個全域模型裡（相對於「只挑一個贏家」的
    做法）。這也是之後 ERR/LFR 要用到的底層聚合規則：ERR/LFR 決定的是
    「這一輪要不要把某個節點的樹排除在合併範圍外」，實際「怎麼把留下來的樹
    接起來」還是要靠這個函式。

    這是從 Flower 官方的 aggregate() 改的：官方版本靠 num_parallel_tree 這個
    固定超參數（通常是 1）判斷要接幾棵樹，只有在「每個節點每輪只新練 1 棵樹」
    時才會給出正確答案，我們每輪新練 10 棵，直接套用會每輪都只接到同一棵
    舊樹。這裡改成直接數 bst_curr_org 裡實際有幾棵樹，不依賴任何可能不準確
    或可能被節點謊報的欄位。

    完整調查過程見 notes/13a-bagging-baseline.md、notes/16-code-review-guide.md。
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
    # 注意：第一個節點是整批（10 筆）算一次疊代，但因為 bst_prev_org 是
    # None 時上面直接回傳原始 JSON，第一個節點自己送來的 iteration_indptr
    # 其實是「一棵樹一筆」的原始記法——兩種記法混在一起，合併幾輪之後
    # bst.num_boosted_rounds() 回傳的數字會失去意義（不等於樹數，也不等於
    # 聯邦輪數）。已確認這不影響下面 client.py 續練時用來切樹的邏輯，但如果
    # 之後有新程式碼想拿這個數字做其他判斷，要先重新檢查。
    iteration_indptr = gbtree_prev["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + num_new_trees)

    for tree_count in range(num_new_trees):
        # 節點送來的 id 是從 0 起算的本地編號（這一批 10 棵樹各自是第
        # 0~9 棵），這裡改成接在 bst_prev 現有樹後面的全域序號。
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        gbtree_prev["trees"].append(trees_curr[tree_count])
        # 固定寫 0，不是讀節點送來的 tree_info——沒有反過來驗證過節點端
        # 訓練出來的 tree_info 是否真的每棵都是 0，是合理推論，不是逐值
        # 核對過的事實。
        gbtree_prev["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None):
        self.num_clients = num_clients
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.latest_model_path = os.path.join(self.model_dir, "global_model_latest.ubj")

        # 直接報錯中止，取代舊版那個靜默的 fallback（self.dval = None ->
        # aggregate_fit() 改成相信 fit_res.metrics["accuracy"]，一個 client
        # 自己算出來、伺服器完全沒有交叉驗證的數字）。惡意 client 只要回報
        # accuracy=1.0，不需要真的訓練，就能在 max() 底下每輪都贏。這個專題
        # 的名字裡有「零信任」三個字，那個 fallback 正是這個精神的具體反例。
        # 完整攻擊情境見 notes/00-findings.md 發現 18。
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

    # zero_division=0 代表某個類別在 y_true 或預測值裡完全缺席時（例如
    # 驗證集或 client 本地測試切分剛好只有單一標籤），precision/recall/F1
    # 會回傳 0.0（不是警告或例外）——這個邊界情況目前還沒有在真實資料上被
    # 真正觸發過，沒有實測驗證。完整驗證紀錄見 notes/12-baseline.md。
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
                # reported_client_id 在這個實驗裡只用來記錄／識別身分
                # （5 個 client 目前都是誠實的）——完全不用於評分或信任判斷，
                # 所以不會重新打開 commit ff82d04 移除的那個自報 fallback 漏洞。
                # 惡意 client 依然可以在這個欄位上說謊；如果任務 13 需要這個
                # 對應關係在有攻擊時也可信，不能依賴這個欄位。
                reported_client_id = (fit_res.metrics or {}).get("client_id", "?")
                print(
                    f"[Server Eval] Client {client_proxy.cid} (client_id={reported_client_id}) "
                    f"Accuracy: {server_metrics['accuracy']:.4f} "
                    f"Precision: {server_metrics['precision']:.4f} "
                    f"Recall: {server_metrics['recall']:.4f} "
                    f"F1: {server_metrics['f1']:.4f}"
                )
                payloads.append((payload, server_metrics, client_proxy.cid, reported_client_id))

        # 排序依據從 accuracy 換成 F1：判斷空間跟「永遠回答攻擊」這個地板分
        # 相差約 19.6 個 F1 百分點，遠高於雜訊範圍，足夠拿來分高下。完整
        # 推導見 notes/11-dataset-check.md、notes/12-baseline.md。
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
    """每一輪把所有節點新練的樹全部接進全域模型，沒有任何節點被丟掉。

    跟 XGBoostStrategy（贏者全拿）的差別：XGBoostStrategy 的 aggregate_fit()
    用 max() 在所有節點的模型裡只挑一個分數最高的當全域模型，其餘節點這一輪
    等於白練；這個策略改成把每個節點的新樹全部接在一起（透過
    aggregate_bagging_verified()，見上方函式），全域模型的樹會一輪一輪
    累積增加。

    這裡多一層傳輸格式轉換：合併用的 aggregate_bagging_verified() 只能處理
    JSON 格式的模型（用 json.loads() 解析），但這個專案節點之間實際傳輸模型
    用的 tensor_type="xgboost-ubj" 是 UBJ 二進位格式，不是 JSON。所以
    aggregate_fit() 裡每收到一個節點的模型，要先轉成 JSON 才能合併
    （bst.save_raw("json")），全部節點都合併完之後，再把結果轉回 UBJ
    （bst.save_raw("ubj")）才能送給下一輪的節點。轉回 UBJ 是必要的、不能省略：
    節點載入模型時是照檔名副檔名 ".ubj" 判斷格式的，如果檔案內容其實是
    JSON，會直接載入失敗。

    已經實測驗證過這個 UBJ→JSON→UBJ 的往返轉換不會遺失任何資訊（轉換前後
    模型的預測結果逐筆相同）。

    完整調查過程見 notes/13a-bagging-baseline.md。
    """

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 leaf_scale: float = 1.0):
        super().__init__(model_dir, num_clients, val_data_path)
        self.global_model_json: Optional[bytes] = None
        # 做成可設定的參數，不寫死成 1/節點數：後者只是一個假設，掃描過
        # 幾個候選值後發現不是最好的——見 notes/13a-leaf-scale-fix.md。
        self.leaf_scale = leaf_scale

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """把這一輪收到的每個節點的新樹，依序縮放、合併，變成新的全域模型。

        Flower 每一輪呼叫一次這個方法。results 是這一輪所有節點回傳的訓練
        結果；每個節點的 FitRes.parameters 裡裝的不是完整模型，是這個節點
        這一輪新練的樹（UBJ bytes，見 client.py 的 _parameters_from_new_trees()）。

        怎麼做：對 results 裡每一個節點的 payload，先轉成 JSON（因為
        aggregate_bagging_verified() 只認 JSON），印出這個節點實際送來幾棵
        樹（直接從 payload 數出來，不採信任何節點自己宣稱的數字），呼叫
        scale_leaf_values() 縮放這批新樹的葉節點值，再呼叫
        aggregate_bagging_verified() 接進正在累積的全域模型 merged_json。
        全部節點都處理完之後，把最終結果轉回 UBJ，存檔、在伺服器驗證集上
        評估、回傳給 Flower。

        沒有對單一節點的 payload 個別評估分數（不像贏者全拿策略那樣印出
        每個節點的分數）：因為這裡的 payload 只是一段樹的片段，不是完整
        模型，脫離它接續訓練的那個整體，單獨拿去預測沒有意義。
        """
        if not results:
            print(f"[Warning] Round {server_round} has no fit results to aggregate.")
            return None, {}

        if failures:
            print(f"[Warning] Round {server_round} had {len(failures)} client failure(s): {failures}")

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

            # 合併前先縮放「這個」節點的新樹批次——在伺服器端做，不在
            # 節點端做，這樣惡意節點就不能單純不套用縮放來規避（零信任，
            # 跟 aggregate_bagging_verified() 不採信任何節點宣稱的樹數量
            # 是同一個道理）。
            payload_json = scale_leaf_values(payload_json, self.leaf_scale)

            merged_json = aggregate_bagging_verified(merged_json, payload_json)

        self.global_model_json = merged_json

        # 把合併後的模型轉回真正的 UBJ bytes 才能傳輸——為什麼這個往返轉換
        # 是必要的、不是可有可無，見 class docstring。
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