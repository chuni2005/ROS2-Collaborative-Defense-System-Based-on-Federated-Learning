"""把 5 個節點的訓練資料集中起來訓練一個模型，當作「架構問題還是資料
問題」的診斷對照組。

原理：聯邦架構下每個節點只看得到自己的資料，如果某個攻擊類型測不到，
有兩種可能：資料裡真的沒有足夠訊號，或是資料裡有訊號、只是現有架構用
不到。把全部節點的資料集中起來訓練同一個模型，如果集中式訓練測得到、
聯邦測不到，就能確認是架構的問題，不是資料的問題。

輸入：split_data/chunk_1..5.csv 全部（不是每個節點各自的 60% 訓練切分，
這樣才跟原本比對用的方法論一致）。
輸出：一個集中式訓練出來的模型（model_centralized_reseed.ubj），以及它在
chunk_6.csv 上的整體指標跟逐攻擊類型 recall／observe 假陽性率
（results/centralized_reseed_recall.txt）。

怎麼做：把 5 個節點的資料合併成一份，用跟 client.py 完全一樣的超參數
（objective=binary:logistic、eta=0.1、max_depth=5、tree_method=hist）
訓練 100 棵樹（對齊聯邦 baseline 10 輪 × 每輪 10 棵樹的總樹數），在
chunk_6.csv 上評估，用跟 analyze_recall_by_attack.py 一樣的方式拆解
逐攻擊類型的表現。

為什麼需要它：這不是要拿來取代聯邦架構的方案——集中式訓練出來的模型
precision／假陽性率遠比聯邦版差，只是拿來當診斷工具，隔離「樣本太少」跟
「架構限制」這兩個可能成因。

完整調查過程見 notes/12-baseline.md。
"""
import os

import pandas as pd
import xgboost as xgb

import server as srv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
SPLIT_DIR = os.path.join(BASE_DIR, "split_data")
NUM_CLIENTS = 5
NUM_BOOST_ROUND = 100

PARAMS = {
    "objective": "binary:logistic", "eta": 0.1, "max_depth": 5,
    "eval_metric": ["logloss"], "tree_method": "hist",
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[Info] Loading and pooling chunk_1..5.csv...")
    dfs = []
    for i in range(1, NUM_CLIENTS + 1):
        df = pd.read_csv(os.path.join(SPLIT_DIR, f"chunk_{i}.csv"), low_memory=False)
        dfs.append(df)
    df_train = pd.concat(dfs, ignore_index=True)
    print(f"[Info] Pooled training rows: {len(df_train)}")

    df_train_p = srv.preprocess_data(df_train)
    X_train = df_train_p.iloc[:, :-1]
    y_train = df_train_p.iloc[:, -1]
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=X_train.columns.tolist())

    print(f"[Info] Training centralized model, num_boost_round={NUM_BOOST_ROUND}...")
    bst = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)

    model_path = os.path.join(BASE_DIR, "model_centralized_reseed.ubj")
    bst.save_model(model_path)
    print(f"[Info] Saved centralized model to {model_path}")

    val_path = os.path.join(SPLIT_DIR, "chunk_6.csv")
    df_val_raw = pd.read_csv(val_path)
    attack_labels = df_val_raw["attack"].copy()
    df_val_p = srv.preprocess_data(df_val_raw.copy())
    X_val = df_val_p.iloc[:, :-1]
    y_val = df_val_p.iloc[:, -1]
    dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=X_val.columns.tolist())

    probs = bst.predict(dval)
    preds = (probs > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, pos_label=srv.POSITIVE_CLASS, zero_division=0)
    rec = recall_score(y_val, preds, pos_label=srv.POSITIVE_CLASS, zero_division=0)
    f1 = f1_score(y_val, preds, pos_label=srv.POSITIVE_CLASS, zero_division=0)
    print(f"\noverall: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    lines = [f"model={model_path}", f"n_trees={bst.num_boosted_rounds()}",
             f"overall accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}", ""]
    lines.append(f"{'attack_type':30s} {'n':>8s} {'value':>10s}  metric")
    print(f"{'attack_type':30s} {'n':>8s} {'value':>10s}  metric")
    for atype in attack_labels.unique():
        mask = (attack_labels == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        positive_rate = float(preds[mask].mean())
        metric = "FPR" if atype == "observe" else "recall"
        row = f"{atype:30s} {n:8d} {positive_rate:10.4f}  {metric}"
        print(row)
        lines.append(row)

    out_path = os.path.join(RESULTS_DIR, "centralized_reseed_recall.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Info] wrote {out_path}")


if __name__ == "__main__":
    main()
