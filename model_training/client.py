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
NUM_BOOST_ROUND = 10  # 每個聯邦輪次新增的樹數；bagging 模式下也用這個數字
                      # 切出「這一輪新增的樹」。

# precision/recall/F1 的正類是 1（「攻擊」）——必須跟 server.py 的 POSITIVE_CLASS 一致
POSITIVE_CLASS = 1

# 標籤翻轉攻擊選要翻轉哪些列用的隨機種子基準。跟 train_test_split 的
# random_state=client_id 分開算，避免「選哪些列翻轉」跟「怎麼切
# 訓練/驗證/測試」共用同一組隨機數狀態、彼此影響。
ATTACK_SEED_BASE = 10000


def flip_labels(label, client_id, rate):
    """標籤翻轉攻擊：把 label 中 rate 比例的樣本標籤反轉（0 和 1 互換），
    回傳新 Series，不修改原始物件。用 ATTACK_SEED_BASE + client_id 當種子
    抽出 round(len(label) * rate) 個位置做 1 - 原值，固定種子確保同一
    client_id、同一 rate 每次重跑選到的列都相同。
    """
    label = label.copy()
    rng = np.random.RandomState(ATTACK_SEED_BASE + client_id)
    n = len(label)
    n_flip = int(round(n * rate))
    flip_pos = rng.choice(n, size=n_flip, replace=False)
    label.iloc[flip_pos] = 1 - label.iloc[flip_pos]
    return label


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


def load_local_data(client_id, data_path, eval_data_path=None, malicious=False, label_flip_rate=1.0):
    """讀取一個節點的本地 CSV，依 client_id 當隨機種子切成訓練:驗證:測試
    = 60:20:20 三份 DMatrix。malicious=True 時，在切分之前就對整份本地
    標籤呼叫 flip_labels()——因此三份切分的標籤都是翻轉後的結果，這個
    節點「本來就是」一份有問題的資料，不是只在送出模型的那一刻造假。
    eval_data_path 目前未使用，是保留給未來外部評估集的參數，永遠回傳
    None。標籤只有單一類別或筆數太少時（can_stratify 系列檢查）自動退化
    成不分層抽樣，避免 train_test_split 直接拋例外。

    回傳 9 元組 (dtrain, dval, dtest_local, deval_external, y_eval_external,
    訓練筆數, 本地測試筆數, 本地測試真實標籤)；deval_external/
    y_eval_external 固定是 None。
    """
    print(f"\n[Info] [Client {client_id}] Loading local training data: {data_path}")

    df_train_raw = pd.read_csv(data_path, low_memory=False)
    df_train_cleaned = preprocess_data(df_train_raw)

    label = df_train_cleaned['attack']
    features = df_train_cleaned.drop(['attack'], axis=1)
    train_feature_names = features.columns.tolist()

    if malicious and label_flip_rate > 0:
        pre_flip_counts = label.value_counts()
        label = flip_labels(label, client_id, label_flip_rate)
        print(
            f"[Attack] [Client {client_id}] Label-flipping attack ACTIVE "
            f"(rate={label_flip_rate}). Before:\n{pre_flip_counts}\nAfter:\n{label.value_counts()}"
        )

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
    """Flower 的 client 端實作，包一個 XGBoost Booster，負責本地訓練、
    序列化模型給伺服器、以及在自己的本地測試集上評估。
    """

    def __init__(self, client_id, data_path, eval_data_path=None, aggregation="winner",
                 malicious=False, label_flip_rate=1.0):
        """載入本地資料（load_local_data()）、固定訓練超參數
        （objective=binary:logistic、eta=0.1、max_depth=5、tree_method=hist）。
        aggregation 決定 fit() 送整個模型還是只送新樹（見 _parameters_from_new_trees()）。
        """
        self.client_id = client_id
        self.aggregation = aggregation
        self.malicious = malicious
        self.label_flip_rate = label_flip_rate

        self.dtrain, self.dval, self.dtest_local, self.deval_ext, self.y_ext, \
            self.num_train, self.num_test, self.y_test_local = \
            load_local_data(client_id, data_path, eval_data_path, malicious, label_flip_rate)

        self.bst = None
        self.current_round = 0
        self.model_load_failures = 0
        self.params = {
            "objective": "binary:logistic", "eta": 0.1, "max_depth": 5,
            "eval_metric": ["logloss"], "tree_method": "hist"
        }

    def _load_model_from_bytes(self, model_bytes):
        """把伺服器傳來的模型 bytes 還原成 XGBoost Booster。Booster.load_model()
        只接受檔案路徑、沒有從記憶體 bytes 讀取的 API，所以先寫暫存檔再讀回、
        刪除。model_bytes 為空或載入失敗都回傳 None（失敗會記錄次數），不拋例外。
        """
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
        """把 self.bst 存成 bytes 供 Parameters 裝載。跟 _load_model_from_bytes()
        對稱：save_model() 只能存檔案，所以先存暫存檔再讀回、刪除。self.bst
        為 None 時回傳 b""。
        """
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
        """印出目前模型依 gain 排序前 15 個特徵，並存成
        output_models/client_{id}_round_{round}.ubj。

        注意：下面的 `if self.bst is None` 檢查寫在 `get_score()` 之後，
        self.bst 真的是 None 時會先在 get_score() 拋 AttributeError，檢查
        永遠不會生效——目前唯一呼叫點在 fit() 訓練完之後，尚未實際觸發過。
        """
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
        """把伺服器傳來的 Parameters 還原成 self.bst；parameters 是 None
        或沒有 tensors 時，self.bst 設回 None（代表要從空模型開始練）。
        """
        if parameters is not None and parameters.tensors:
            model_bytes = bytes(parameters.tensors[0])
            self.bst = self._load_model_from_bytes(model_bytes)
        else:
            self.bst = None

    def _parameters_from_booster(self) -> Parameters:
        """把 self.bst 整個模型包成 Parameters 物件
        （tensor_type="xgboost-ubj"），winner 模式下用來把累積至今的
        完整模型送給伺服器。
        """
        model_bytes = self._serialize_model_to_bytes()
        tensors = [model_bytes] if model_bytes else []
        return Parameters(tensors=tensors, tensor_type=TENSOR_TYPE)

    def _parameters_from_new_trees(self) -> Parameters:
        """只把這個節點這一輪新練的樹切出來包成 Parameters，不是整個模型。

        XGBoost 的 Booster 支援切片 booster[a:b]，取出第 a 到 b 個訓練
        疊代組成新的 Booster；每次 xgb.train(..., num_boost_round=N,
        xgb_model=舊模型) 都會在舊模型後面剛好新增 N 個疊代（本專案設定下
        一個疊代就是一棵樹）。所以用 self.bst[total_rounds-NUM_BOOST_ROUND
        : total_rounds] 切出來的，剛好就是這一輪新練出的樹，不多不少；序列化
        時一樣先存暫存檔再讀回 bytes（XGBoost 沒有記憶體序列化 API）。

        只在 bagging 模式使用：如果每輪都送整個累積模型，伺服器沒辦法知道
        這次新增的是哪幾棵，只能猜或相信節點自報；只送新樹，伺服器直接數
        收到幾棵就知道要接幾棵。
        """
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

    def _evaluate_global_on_local_test(self):
        """印出目前 self.bst（收到的全域模型）在這個節點自己本地測試集上的
        accuracy/precision/recall/F1，純觀察用途，結果不會被送回伺服器或
        用於任何判斷。self.bst 是 None 時只印警告、不計算。
        """
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

    # zero_division=0：本地測試切分若剛好只剩單一標籤，precision/recall/F1
    # 回傳 0.0 而不拋例外——這個邊界情況目前未在實際資料上驗證過。
    def _evaluate_local_test(self):
        """算 self.bst 在本地測試集上的 logloss/accuracy/precision/recall/f1，
        回傳 dict。logloss 手動算交叉熵（XGBoost 的 eval_metric 只在訓練時
        的 evals 列表起作用，訓練後另外對測試集算要自己實作），機率夾在
        [1e-7, 1-1e-7] 避免 log(0)。失敗或 self.bst 為 None 回傳全 0
        （logloss=1.0）的預設 dict，不拋例外。
        """
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
        """Flower 的 GetParametersIns/Res 介面方法：回傳 self.bst 目前的
        模型（透過 _parameters_from_booster()），供伺服器在訓練開始前取得
        初始模型。
        """
        parameters = self._parameters_from_booster()
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Flower 每輪呼叫一次：載入全域模型、續練 NUM_BOOST_ROUND 棵新樹、
        依 aggregation 模式回傳整個模型或只回傳新樹（metrics 裡的
        accuracy/precision/recall/f1/logloss 是本地測試集算出來的）。
        """
        # --- 還原全域模型、評估更新前的表現 ---
        self.current_round += 1
        print(f"\n[Client {self.client_id}] Recieve Server Message - {self.current_round} Round Fit Start")

        self._set_booster_from_parameters(ins.parameters)
        self._evaluate_global_on_local_test()

        # --- 用本地資料續練 NUM_BOOST_ROUND 棵新樹 ---
        print(f"[Info] [Client {self.client_id}] Training...")
        self.bst = xgb.train(
            self.params, self.dtrain, num_boost_round=NUM_BOOST_ROUND,
            evals=[(self.dtrain, "train"), (self.dval, "val")],
            xgb_model=self.bst, verbose_eval=False
        )

        # --- 在本地測試集上評估、存檔留底 ---
        local_metrics = self._evaluate_local_test()
        self._save_model_artifact()

        # --- 依 aggregation 模式組裝要送出的 Parameters ---
        # winner 模式送出整個累積的模型（伺服器挑一個贏家的完整模型當下一輪
        # 的起點）。bagging 模式只送這一輪新增的樹——見 _parameters_from_new_trees()。
        if self.aggregation == "bagging":
            fit_parameters = self._parameters_from_new_trees()
        else:
            fit_parameters = self._parameters_from_booster()

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=fit_parameters,
            num_examples=self.num_train,
            metrics={
                # 純觀察用途，方便伺服器 log 對照——client 自報，伺服器不
                # 驗證真偽，也不用於任何評分或篩選判斷（server.py 的
                # aggregate_fit() 只依自己算出來的 f1 排序）。
                "client_id": self.client_id,
                # 同上，純觀察用途，client 自報不驗證。
                "malicious": self.malicious,
                "accuracy": local_metrics["accuracy"],
                "precision": local_metrics["precision"],
                "recall": local_metrics["recall"],
                "f1": local_metrics["f1"],
                "logloss": local_metrics["logloss"],
                "model_load_failures": self.model_load_failures,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Flower 的 EvaluateIns/Res 介面方法：跑一次本地測試集評估，回傳
        loss（logloss）與 accuracy/precision/recall/f1。self.bst 還原失敗時
        回傳 loss=1.0、指標全 0，不拋例外。

        目前沒有被實際呼叫：server.py 的 fraction_evaluate=0.0、
        min_evaluate_clients=0（XGBoostStrategy.__init__()），Flower 不會
        對任何節點呼叫這個方法。
        """
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
    parser.add_argument("--malicious", action="store_true",
                        help="Turn this client into a label-flipping attacker (see flip_labels()). "
                             "Off by default; the honest majority never pass this flag.")
    parser.add_argument("--label_flip_rate", type=float, default=1.0,
                        help="Only used when --malicious is set. Fraction of this client's local "
                             "labels to flip (1.0 = flip all, the default full-strength attack).")
    args = parser.parse_args()

    print(f"Flower Client {args.client_id} running ...")
    fl.client.start_client(
        server_address=args.server_address,
        client=XGBoostClient(
            args.client_id, args.data_path, aggregation=args.aggregation,
            malicious=args.malicious, label_flip_rate=args.label_flip_rate,
        )
    )