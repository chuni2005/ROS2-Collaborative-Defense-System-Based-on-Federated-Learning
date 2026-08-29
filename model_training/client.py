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

# precision/recall/F1 的正類是 1（「攻擊」）——必須跟 server.py 的
# POSITIVE_CLASS 一致。見 notes/11-dataset-check.md。
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

    def _parameters_from_new_trees(self) -> Parameters:
        """只把這個節點這一輪新練的樹切出來、包成 Parameters，不是整個模型。

        原理：XGBoost 的 Booster 支援切片語法 booster[a:b]，取出「第 a 到 b
        次訓練疊代對應的樹」組成一個新的、獨立的 Booster。每次呼叫
        xgb.train(..., num_boost_round=N, xgb_model=舊模型) 都會在舊模型
        後面剛好新增 N 個疊代（這個專案的設定下，一次疊代就是一棵樹）。所以
        「總疊代數 − 這次新增的疊代數」到「總疊代數」這一段切片，剛好就是
        這一輪新練出來的那些樹，不多也不少。

        輸入：self.bst — 這個節點目前的完整模型（上一輪收到的全域模型 +
              這一輪剛練好的 NUM_BOOST_ROUND 棵新樹）
        輸出：Parameters 物件，裡面只裝這一輪新增的 NUM_BOOST_ROUND 棵樹，
              序列化成 UBJ bytes（不是 self.bst 整個模型）

        怎麼做：total_rounds = self.bst.num_boosted_rounds() 拿到目前總共
        練了幾個疊代；用切片 self.bst[total_rounds-NUM_BOOST_ROUND :
        total_rounds] 取出最後 NUM_BOOST_ROUND 個疊代，變成一個新的
        Booster；因為 XGBoost 沒有直接序列化到記憶體 bytes 的 API，這裡先
        存成暫存檔、讀回 bytes、再刪掉暫存檔。

        為什麼需要它：只在 bagging 模式使用。如果每輪都送整個累積的模型，
        伺服器合併時沒辦法知道「這次新增的是哪幾棵」，只能用不可靠的欄位去
        猜（這正是直接套用 Flower 官方 aggregate() 會出問題的原因，見
        aggregate_bagging_verified() 的 docstring）。只送這一輪新練的樹，
        伺服器直接數收到幾棵就知道要接幾棵，不需要猜、也不用相信節點自己
        宣稱新增了幾棵。

        完整調查過程見 notes/00-findings.md、notes/13a-bagging-baseline.md。
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

    # zero_division=0：某個 client 的本地測試切分（自己那份 chunk 的 20%）
    # 剛好只有單一標籤時，precision/recall/F1 會回傳 0.0，不會拋出例外或
    # 警告——這個邊界情況目前還沒有在真實資料上被觸發過，所以這個預設值
    # 沒有拿我們實際的 chunk 大小驗證過。
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
                # 純觀察用途——讓伺服器的 log 能把 Flower 內部的 cid 對應到
                # 我們自己的 1-5 編號。從不用於評分或信任判斷（server.py 的
                # aggregate_fit() 只依自己算出來的 f1 排序），所以說謊的
                # client 沒辦法像被移除的那個 accuracy 自報 fallback 一樣
                # 利用這個欄位——完整理由見 server.py 裡 reported_client_id
                # 那段註解。
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