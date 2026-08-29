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
    """Scale every leaf node's value in a serialized booster's trees by w.

    中文導讀：
    吃什麼：`model_json` 是「一個 client 這一輪新增的樹」序列化成的 JSON bytes——
        bagging 模式下 client 每輪只送新增的 `NUM_BOOST_ROUND=10` 棵樹，不是完整
        模型（見 client.py 的 `_parameters_from_new_trees()`）；`w` 是縮放係數
        （例如 0.5，實驗上已定案，見 notes/13a-leaf-scale-fix.md）。
    吐什麼：結構完全相同的 JSON bytes——樹的數量、分裂結構都不變，只有每棵樹
        葉節點的數值被乘上 `w`。
    中間轉換：對每一棵樹，找出所有「葉節點」（`left_children[i] == -1` 的節點），
        把該節點的 `base_weights[i]` 與 `split_conditions[i]` 兩個欄位都乘上 `w`。
    在流程裡的位置：`server.py` 的 `aggregate_fit()`（bagging 策略）裡，在呼叫
        `aggregate_bagging_verified()` 合併「之前」，對每個 client 這一輪送來的
        樹批次分別呼叫一次，縮放完才送進合併函式——所以縮放的對象永遠是「單一
        client 這一輪的新樹」，不是已經合併好的全域模型。

    修的是 notes/13a-bagging-baseline.md 發現的 margin 暴衝問題：5 個 client
    每輪各自獨立對「同一個」共享全域模型新練 10 棵樹，aggregate_bagging_verified()
    把 5 份修正加總（不是取平均）——等同把學習率放大 5 倍，逐輪疊加下去，直到
    margin 大到把 sigmoid 完全飽和（round 1：[-8, 7]；round 10：
    [-14214, 2283]，細節見 notes/13a）。

    插入點依據 notes/00-findings.md 發現 3、notes/02 第 4 節的推論：因為
    XGBoost 的預測值是 base_score 加上每棵樹選中的葉節點值加總，在合併進全域
    模型「之前」，把一個 client 這批新樹的所有葉節點值乘上 w，就會讓這批樹
    對整體預測的貢獻剛好縮放 w 倍——這件事已經實測驗證過（葉節點的值同時存在
    base_weights[i] 跟 split_conditions[i] 這兩個欄位，也就是 left_children[i]
    == -1 的節點；對一個真實的 client_1_round_1.ubj 模型兩個欄位都乘 w=0.3，
    量到的邊際貢獻比值是 0.29999995~0.30000004，跟理論值的最大誤差只有
    9e-8——這解決了 notes/02 標記為「[推測，待驗證]」的葉值公式）。兩個欄位
    一起縮放、不是只改一個，是為了讓 client 之後如果拿這個模型繼續訓練
    （xgb_model=...），看到的樹內部兩個欄位還是一致的，不會出現
    base_weights 跟 split_conditions 對不上的情況。
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

    中文導讀：
    吃什麼：`bst_prev_org` 是「這一輪目前為止已經合併好的全域模型」，JSON bytes；
        這一輪第一次呼叫時是 `None`（還沒有任何 client 合併進來）。`bst_curr_org`
        是「這個 client 這一輪新增的樹」，JSON bytes，呼叫這個函式之前已經被
        `scale_leaf_values()` 縮放過（見 `aggregate_fit()` 裡的呼叫順序）。
    吐什麼：合併後的全域模型，JSON bytes——`bst_curr_org` 的樹全部接到
        `bst_prev_org` 的樹陣列後面，樹計數、索引指標等記帳欄位同步更新。
    中間轉換：**不重新訓練、不改變任何一棵樹本身的結構**，純粹是陣列串接——
        把 `bst_curr_org["trees"]` 依序 append 到 `bst_prev_org["trees"]` 後面，
        重新編號 `id`，更新 `num_trees`／`iteration_indptr`／`tree_info`。
    在流程裡的位置：`server.py` 的 `XGBoostBaggingStrategy.aggregate_fit()`
        裡，對這一輪收到的每個 client payload 依序呼叫一次（5 個 client 就呼叫
        5 次），逐一疊加成這一輪的全域模型。**這個函式就是 ERR/LFR 需要的「底層
        聚合規則」**——ERR/LFR 本身只是站在這一層之上，決定「這一輪要不要把某個
        client 的樹排除在合併範圍外」，沒有一個能正確合併「保留下來的那些
        client」的底層規則，ERR/LFR 沒有東西可以聚合。目前分支還沒有這層排除
        邏輯，`run_loo_impact.py` 是用「整批排除某個 client、重跑整個 10 輪」的
        方式模擬這件事，不是即時的逐輪排除。

    這是從 Flower 官方 aggregate()（flwr/server/strategy/fedxgb_bagging.py:
    118-154）改的，但修了一件事：官方版本是讀 bst_curr_org 裡的
    gbtree_model_param.num_parallel_tree 來決定要接幾棵樹——這是一個固定的
    XGBoost 超參數（每次 boosting 疊代要平行訓練幾棵樹，正常是 1），不是
    「這次送來的東西裡有幾棵新樹」。只有當 client 剛好每輪只訓練
    num_parallel_tree（=1）棵新樹時，這個欄位才會剛好給出正確答案——這正是
    Flower 自己的 bagging 範例的做法（local-epochs=1）。我們的 client.py
    每輪訓練 NUM_BOOST_ROUND=10 棵樹，這個欄位不管實際新增了幾棵都恆為 1——
    直接拿來用，會讓每一輪都靜默地只合併到同一棵過時的第一棵樹，不是每輪
    真正的新樹。詳見 notes/00-findings.md 發現 21 與 notes/13a-bagging-baseline.md
    記錄的完整發現過程。

    修法：不讀任何 client 自己宣稱或聲明的數量。client.py 現在只送出這一輪
    新增的樹（見 client.py 的 _parameters_from_new_trees()），所以要接的
    樹數就是 len(trees in bst_curr_org)——直接從實際收到的 payload 數出來，
    不採信任何 client 可能謊報的欄位。這跟 ff82d04 移除的那個 fallback
    是同一種「不相信節點自己的宣稱，親自查證 payload 本身」的原則。
    """
    # 這一輪第一個被處理的 client：還沒有「前一個 client 合併好的模型」可以接，
    # 這個 client 的樹本身就是這一輪的起點，直接回傳，不用跑下面的合併邏輯。
    if not bst_prev_org:
        return bst_curr_org

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    gbtree_prev = bst_prev["learner"]["gradient_booster"]["model"]
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    tree_num_prev = int(gbtree_prev["gbtree_model_param"]["num_trees"])
    num_new_trees = len(trees_curr)

    gbtree_prev["gbtree_model_param"]["num_trees"] = str(tree_num_prev + num_new_trees)
    # iteration_indptr 記法在合併多輪之後會變得不一致（第一個 client 保留了
    # 「每棵樹一筆」的原始記法，後續每個 client 合併時卻是「一整批新樹算一筆」）,
    # 導致 bst.num_boosted_rounds() 之後回傳的數字沒有語意（不等於樹數也不等於
    # 聯邦輪數）。已確認這不影響 client.py 續練時的樹切片邏輯（見
    # notes/16-code-review-guide.md Part B 第 5 項的驗證），但如果之後有新程式碼
    # 想拿 num_boosted_rounds() 的數字做其他判斷，要先重新檢查。
    iteration_indptr = gbtree_prev["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + num_new_trees)

    for tree_count in range(num_new_trees):
        # 重新編號 id：client 自己送來的樹，id 是從 0 起算的本地編號（這一批
        # 10 棵樹各自是第 0~9 棵），接進全域模型後必須改成在「整個模型」裡的
        # 全域序號，否則會跟 bst_prev 裡既有的樹 id 重複。
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        gbtree_prev["trees"].append(trees_curr[tree_count])
        # 固定寫 0，不是讀 client 送來的 tree_info：這個欄位是 XGBoost 給
        # 多輸出模型（例如多分類）用來標記「這棵樹屬於第幾個輸出」的，這個
        # 專案是單一輸出的二元分類（binary:logistic），永遠只有一組，寫 0
        # 對這個用途是安全的——但沒有反過來驗證過 client 端訓練出來的
        # tree_info 是否真的每棵都是 0，是合理推論，不是逐值核對過的事實。
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
        # accuracy=1.0，不需要真的訓練，就能在 max() 底下每輪都贏——完整的
        # 攻擊情境見 notes/00-findings.md 發現 18。這個專題的名字裡有「零信任」
        # 三個字，那個 fallback 正是這個精神的具體反例。
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

    # 已用 notes/12-baseline.md 的真實資料驗證過（兩次完整的 10 輪執行，
    # baseline_run1/run2）。zero_division=0 代表某個類別在 y_true 或預測值
    # 裡完全缺席時（例如驗證集或 client 本地測試切分剛好只有單一標籤），
    # precision/recall/F1 會回傳 0.0（不是警告或例外）——這個邊界情況目前
    # 還沒有在真實資料上被真正觸發過，沒有實測驗證。
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

        # 排序依據從 accuracy 換成 F1——見 notes/11-dataset-check.md「誠實模型的
        # accuracy 基準線」一節，以及 notes/12-baseline.md（已驗證：判斷空間跟
        # 「永遠回答攻擊」這個地板分相差約 19.6 個 F1 百分點，遠高於雜訊範圍）。
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
    """notes/13a-bagging-baseline.md 的 bagging 聚合 baseline。

    aggregate_fit() 每一輪把每個 client 的新樹接進正在跑的全域模型——沒有
    任何一個 client 會被丟掉，跟 XGBoostStrategy 用 max() 只留一個贏家不同。

    相較於一開始直接套用 Flower 官方 aggregate() 的做法，有兩件事必須修正
    （完整調查過程見 notes/00-findings.md 發現 21、notes/13a-bagging-baseline.md）：

    1. 合併邏輯：Flower 官方的 aggregate() 讀 gbtree_model_param.
       num_parallel_tree 來決定要接幾棵樹——這是一個固定的 XGBoost 超參數
       （正常是 1），不是「這次送來的東西裡有幾棵新樹」。這個專案的
       NUM_BOOST_ROUND=10，這個欄位不管實際新增了幾棵永遠是 1，直接拿來用
       會讓每一輪都靜默地合併到同一棵過時的舊樹。上面的
       aggregate_bagging_verified() 改成直接數實際 payload 裡的 len(trees)——
       這是從 payload 本身的結構得到的事實，不採信任何 client 宣稱的值，
       也不採信反映不出實際情況的超參數。
    2. 傳輸格式：client.py 現在只送出這一輪新增的樹（見
       _parameters_from_new_trees()），不是送整個累積的模型——這是第 1 點
       「有幾棵新樹」這個問題本身要有明確定義的前提。

    Flower 的 aggregate() 用 json.loads() 解析模型 bytes，所以 bagging 需要
    JSON 格式，不是這個專案 tensor_type="xgboost-ubj" 實際傳輸用的 UBJ
    bytes。已經實測驗證過：把 UBJ bytes 載入 Booster、再用 save_raw("json")
    重新序列化，轉換是正確的；把合併後的 JSON 結果用 save_raw("ubj") 轉回去，
    也是無損的往返轉換（預測結果完全相同）——這件事在真正依賴它之前，先用一個
    合成模型確認過。轉回 UBJ 這一步是必要的，因為 client.py 的
    _load_model_from_bytes() 是照檔名副檔名 ".ubj" 來載入的，xgboost 的
    load_model(path) 會相信這個副檔名——JSON 內容放進一個叫 ".ubj" 的檔案
    會直接載入失敗（直接測試過：json.cc 解析錯誤），所以傳輸格式必須維持
    真正的 UBJ。
    """

    def __init__(self, model_dir: str, num_clients: int, val_data_path: Optional[str] = None,
                 leaf_scale: float = 1.0):
        super().__init__(model_dir, num_clients, val_data_path)
        self.global_model_json: Optional[bytes] = None
        # 見 scale_leaf_values() docstring 與 notes/13a-bagging-baseline.md
        # 的「未解問題」1——修的是「5 個 client 每輪各自獨立修正，伺服器
        # 加總而非取平均」造成的 margin 暴衝問題。做成可設定的參數，不寫死
        # 1/client 數：後者只是一個假設（跟 NVFlare 的 lr_mode="uniform"
        # 預設值一樣，見 notes/03、notes/00-findings.md 發現 5），對這個
        # 專案 num_boost_round=10-每輪 的設定來說不是已經證實的事實。
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

        # 現在每個 client 只送出這一輪新增的樹（見 client.py 的
        # _parameters_from_new_trees()），所以這裡的 payload 不是一個完整的
        # 模型——單獨拿去對 chunk_6 評估，不代表「這個 client 的模型品質」
        # （一段樹的片段本身，脫離它接續訓練的那個整體，輸出的原始值不是有
        # 意義的機率），所以贏者全拿策略裡那個逐 client 評估的步驟，這裡是
        # 刻意拿掉的。改成印出每個 payload 裡實際找到的樹數——直接從 payload
        # 本身數出來，不採信任何 client 自己宣稱的數字——這樣如果有 client
        # 送來的樹數比預期多或少，log 裡看得出來。
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

            # 合併前先縮放「這個」client 的新樹批次——在伺服器端做，不在
            # client 端做，這樣惡意 client 就不能單純不套用縮放來規避
            # （零信任，跟下面 len(trees_curr) 不採信任何 client 宣稱的數量
            # 是同一個道理——見 aggregate_bagging_verified() 的 docstring）。
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