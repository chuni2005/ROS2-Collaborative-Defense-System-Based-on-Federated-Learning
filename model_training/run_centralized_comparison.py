"""把 5 個節點的訓練資料集中起來訓練一個模型，當作「架構問題還是資料問題」
的診斷對照組——不是要取代聯邦架構，純粹是診斷用途。

本檔案唯一需要修改的實驗設定在區塊 1。
"""
# ==================== 區塊 1：實驗設定 ====================
# 這支腳本只跑單一組集中式訓練，會影響輸出的參數都是這裡的模組層級常數與 PARAMS。
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
    """把 5 個節點的訓練資料集中起來訓練一個 100 棵樹的 XGBoost 模型，在
    chunk_6.csv 上評估整體指標與逐攻擊類型表現，寫成
    results/centralized_reseed_recall.txt，當作聯邦架構的診斷對照組。
    各階段見下方區塊 2-5 註解。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ==================== 區塊 2：讀取並合併資料 ====================
    # --- 讀取並合併 5 個節點的訓練資料 ---
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

    # ==================== 區塊 3：訓練模型 ====================
    # --- 用跟 client.py 相同的超參數訓練 ---
    print(f"[Info] Training centralized model, num_boost_round={NUM_BOOST_ROUND}...")
    bst = xgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)

    model_path = os.path.join(BASE_DIR, "outputs", "models", "centralized_reseed.ubj")
    bst.save_model(model_path)
    print(f"[Info] Saved centralized model to {model_path}")

    # ==================== 區塊 4：評估整體指標 ====================
    # --- 在 chunk_6.csv 上評估整體指標 ---
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

    # ==================== 區塊 5：逐攻擊類型計算與寫檔 ====================
    lines = [f"model={model_path}", f"n_trees={bst.num_boosted_rounds()}",
             f"overall accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}", ""]
    # --- 逐攻擊類型計算 recall／FPR 並寫檔 ---
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
