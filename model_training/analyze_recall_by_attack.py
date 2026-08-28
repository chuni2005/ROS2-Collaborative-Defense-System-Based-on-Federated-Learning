"""Per-attack-type recall / false-positive-rate breakdown for a saved model.

Used by run_leaf_scale_sweep.py (see notes/13a-bagging-baseline.md and the
leaf-scale follow-up note) to compare how each leaf_scale value affects
detection of individual attack subtypes on the server's validation chunk
(chunk_6.csv), not just the binarized overall F1.

'observe' (label 0, benign) is reported as false-positive rate (fraction
wrongly predicted as attack); every other attack subtype (label 1) is
reported as recall (fraction correctly predicted as attack). This mirrors
the table format already used in notes/13a-bagging-baseline.md.
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

import server as srv


def main():
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
