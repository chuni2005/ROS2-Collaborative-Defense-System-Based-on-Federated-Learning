"""Centralized-pooling diagnostic comparison, reproduced on the reseeded
split_data (SPLIT_SEED=42). Original methodology and rationale:
notes/12-baseline.md "三種偵測失敗類型的成因" section -- this is a
diagnostic tool to isolate "not enough samples" from "federated
architecture can't use the samples that exist", NOT a proposed
replacement for the federated pipeline (its precision/FPR are far worse,
see below).

Pools chunk_1.csv..chunk_5.csv (the full chunks, not each client's 60%
train split -- matches the original methodology) into a single XGBoost
model trained with the same hyperparameters client.py uses
(objective=binary:logistic, eta=0.1, max_depth=5, tree_method=hist),
num_boost_round=100 to match the federated baseline's cumulative tree
count after 10 rounds x 10 trees/round. Evaluated on chunk_6.csv with
the same per-attack-type recall / observe-FPR breakdown as
analyze_recall_by_attack.py.
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
