"""畫 analyze/results/preds_*.csv 裡每個檔案的 ROC 曲線，疊在同一張圖上比較。

只用 glob 讀 preds_*.csv，不重新跑模型、不碰 test.csv。每個檔案代表一個情境
（scenario），檔名 preds_<scenario>.csv 裡 <scenario> 的部分當圖例名稱；每個
檔案要有 y_true、y_score 兩欄（跟 predictions.parquet 同一套欄位命名）。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load_one(path):
    """讀一個 preds_*.csv，算出 (fpr, tpr, auc)。

    缺欄位、欄位全空、或 y_true 只有一個類別（算不出 ROC）都印警告後回傳
    None，只跳過這個檔案，不中止整支腳本。
    """
    df = pd.read_csv(path)

    for col in ("y_true", "y_score"):
        if col not in df.columns:
            print(f"[警告] {path.name} 缺少欄位 '{col}'，略過這個檔案。")
            return None
        if df[col].isna().all():
            print(f"[警告] {path.name} 的 '{col}' 欄位全部是空值，略過這個檔案。")
            return None

    if df["y_true"].nunique() < 2:
        print(f"[警告] {path.name} 的 y_true 只有一個類別，無法計算 ROC，略過這個檔案。")
        return None

    fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
    auc = roc_auc_score(df["y_true"], df["y_score"])
    return fpr, tpr, auc


def main():
    """畫 ROC 曲線，回傳存檔路徑；沒有可用的 preds_*.csv 就回傳 None。"""
    files = sorted(RESULTS_DIR.glob("preds_*.csv"))
    if not files:
        print(f"[警告] {RESULTS_DIR} 底下找不到任何 preds_*.csv，沒有資料可畫，結束。")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    plotted = 0

    for i, path in enumerate(files):
        scenario = path.stem[len("preds_"):]
        result = load_one(path)
        if result is None:
            continue
        fpr, tpr, auc = result

        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        marker = MARKERS[i % len(MARKERS)]
        # ROC 曲線的點數等於資料裡相異分數的個數，可能有幾千個，每個點都放
        # marker 會糊成一片，所以用 markevery 隔一段距離放一個，只是給黑白
        # 列印一個可辨識的形狀線索，不是要標出每個門檻值。
        markevery = max(1, len(fpr) // 15)

        ax.plot(
            fpr, tpr, color=color, linestyle=linestyle, marker=marker,
            markevery=markevery, label=f"{scenario} (AUC={auc:.3f})",
        )
        plotted += 1

    if plotted == 0:
        print("[警告] 所有 preds_*.csv 都因缺欄位或資料問題被略過，沒有東西可畫，結束。")
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random (y=x)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Info] 圖已存到 {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
