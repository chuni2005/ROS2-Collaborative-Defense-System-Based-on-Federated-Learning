"""載入贏者全拿與 bagging 兩組已訓練好的 baseline 模型（round 10），
在 test_natural.csv 上重算一次整體 F1 與逐攻擊類型 recall，跟原本只在
chunk_6.csv 上量到的數字對照——不重新訓練，只是換一份資料重新推論。

用 XGBoostStrategy 建構子順便帶上 test_natural_data_path，直接呼叫
_evaluate_model_on_server(dataset="test_natural") 拿整體指標，這樣同時
也是 server.py 這次新增的 dataset 參數的一次實際呼叫驗證，不是另外重寫
一套算法。逐攻擊類型 recall 沿用 analyze_recall_by_attack.py 的做法：
用原始（未二元化）的 attack 欄位分組，對每一組算「被判定為攻擊」的比例
——對 observe 這個比例是假陽性率，對其餘每種攻擊類型是 recall。
"""
import tempfile

import pandas as pd
import xgboost as xgb

import server as srv

MODELS = {
    "winner_take_all": "model_baseline_reseed/global_model_round_10.ubj",
    "bagging_leaf_scale_0.5": "model_leaf_scale_1-2/global_model_round_10.ubj",
}


def per_type_recall(bst, df_raw):
    """對同一個模型，用原始 attack 字串欄位分組算每種類型「被判定為攻擊」
    的比例。回傳 [(攻擊類型, 該類型筆數, 比例, "recall"或"FPR"), ...]，
    observe 標成 FPR（假陽性率），其餘攻擊類型標成 recall。
    """
    attack_labels = df_raw["attack"].copy()
    df = srv.preprocess_data(df_raw.copy())
    X = df.iloc[:, :-1]
    dm = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
    preds = (bst.predict(dm) > 0.5).astype(int)

    rows = []
    for atype in attack_labels.unique():
        mask = (attack_labels == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        rate = float(preds[mask].mean())
        metric = "FPR" if atype == "observe" else "recall"
        rows.append((atype, n, rate, metric))
    return rows


def main():
    """對 MODELS 裡的每個模型印出：test_natural 整體 accuracy/precision/
    recall/f1（來自 _evaluate_model_on_server），以及逐攻擊類型 recall/FPR
    （來自 per_type_recall）。只印在終端機，不寫檔——這次是一次性核對，
    不是要建立新的可重跑管線。
    """
    with tempfile.TemporaryDirectory() as scratch_model_dir:
        strategy = srv.XGBoostStrategy(
            model_dir=scratch_model_dir,
            num_clients=5,
            val_data_path="split_data/chunk_6.csv",
            test_natural_data_path="split_data/test_natural.csv",
        )

        df_raw = pd.read_csv("split_data/test_natural.csv")

        for name, path in MODELS.items():
            with open(path, "rb") as f:
                model_bytes = f.read()

            overall = strategy._evaluate_model_on_server(model_bytes, dataset="test_natural")
            bst = xgb.Booster()
            bst.load_model(bytearray(model_bytes))
            rows = per_type_recall(bst, df_raw)

            print(f"=== {name} ({path}) ===")
            print(f"overall on test_natural.csv: accuracy={overall['accuracy']:.4f} "
                  f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
                  f"f1={overall['f1']:.4f}")
            for atype, n, rate, metric in rows:
                print(f"  {atype:30s} n={n:7d} {metric}={rate:.4f}")
            print()


if __name__ == "__main__":
    main()
