"""對一個已經訓練好的模型，拆解出逐攻擊類型的 recall，不是只看整體 F1。

原理：整體 F1／accuracy 是把所有攻擊類型混在一起算的單一數字，如果某幾種
攻擊佔了驗證集大多數樣本，整體指標會被這幾種主導，稀釋掉少數類型測不到
的問題。這支腳本用原始的（未二元化的）attack 欄位分組，逐一計算每種類型
自己的表現，才看得出哪些類型測得到、哪些測不到。

輸入：一個已存檔的模型（.ubj 檔案路徑），以及驗證資料 CSV（預設
chunk_6.csv，需要保留原始的 attack 字串欄位，不能先二元化）。
輸出：印在終端機上的一張表，每種攻擊類型一列，數字是「這個類型裡有多少
比例被模型判定為攻擊」。

怎麼做：用跟 server.py 一樣的 preprocess_data() 把特徵欄位處理成模型看得
懂的格式，但保留原始的 attack 字串欄位另外存起來；模型預測完，對每個
攻擊類型的樣本分別算「預測為攻擊」的比例——對 observe（正常流量）這個
比例代表假陽性率，對其餘每種攻擊類型這個比例就是 recall。

為什麼需要它：被 run_leaf_scale_sweep.py 等腳本呼叫，用來比較不同設定
（例如不同的 leaf_scale）對個別攻擊類型偵測率的影響，不是只看整體 F1
這一個數字。

完整調查過程見 notes/13a-bagging-baseline.md、notes/13a-leaf-scale-fix.md。
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

import server as srv


def main():
    """指令列進入點：載入 --model_path 指定的模型，對 --val_data_path 的
    驗證資料逐攻擊類型計算「被判定為攻擊」的比例（對 observe 這個比例是
    假陽性率 FPR，對其餘攻擊類型是 recall），結果印在終端機（不寫檔，寫
    檔由呼叫端負責，例如 run_leaf_scale_sweep.py 會把這支腳本的 stdout
    導進自己的 recall 檔案）。
    """
    parser = argparse.ArgumentParser(description="Per-attack-type recall/FPR + margin stats for one model.")
    parser.add_argument("--model_path", required=True, help="Path to a .ubj booster file.")
    parser.add_argument("--val_data_path", default="split_data/chunk_6.csv",
                         help="CSV with the original (non-binarized) 'attack' column.")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.val_data_path)
    attack_labels = df_raw["attack"].copy()

    df = srv.preprocess_data(df_raw.copy())
    X = df.iloc[:, :-1]
    feature_names = X.columns.tolist()

    bst = xgb.Booster()
    bst.load_model(args.model_path)
    dm = xgb.DMatrix(X.values, feature_names=feature_names)
    probs = bst.predict(dm)
    margin = bst.predict(dm, output_margin=True)
    preds = (probs > 0.5).astype(int)

    print(f"model={args.model_path}")
    print(f"n_trees={bst.num_boosted_rounds()}")
    print(f"margin_min={margin.min():.4f} margin_max={margin.max():.4f} margin_mean={margin.mean():.4f}")
    print()
    print(f"{'attack_type':30s} {'n':>8s} {'value':>10s}  metric")
    for atype in attack_labels.unique():
        mask = (attack_labels == atype).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        positive_rate = float(preds[mask].mean())
        metric = "FPR" if atype == "observe" else "recall"
        print(f"{atype:30s} {n:8d} {positive_rate:10.4f}  {metric}")


if __name__ == "__main__":
    main()
