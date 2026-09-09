"""畫 analyze/results/predictions.parquet 裡每種 attack_raw 類型被判成攻擊的比例。

只讀 predictions.parquet，不重新跑模型、不碰 test.csv。門檻固定 0.5，跟
client.py/server.py 的 preds_prob > 0.5 用同一個門檻，數字才能互相比較。

這個比例對 observe 跟對其他攻擊類型意義相反：observe 那一條是假陽性率（越低
越好），其餘攻擊類型那幾條是 recall（越高越好）——不能放在同一個顏色裡當成
同一件事比大小，所以 observe 用不同顏色標出來。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.parquet"

OBSERVE_LABEL = "observe"
OBSERVE_COLOR = "#d62728"  # 假陽性率，跟其他攻擊類型的 recall 意義相反，顏色要分開
ATTACK_COLOR = "#1f77b4"


def main():
    """畫每種 attack_raw 的預測比例，回傳 (rate, count) 的表格；沒資料可畫就回傳 None。"""
    if not PREDICTIONS_PATH.exists():
        print(f"[警告] 找不到 {PREDICTIONS_PATH}，沒有資料可畫，結束。")
        return None

    df = pd.read_parquet(PREDICTIONS_PATH)

    required_cols = ["attack_raw", "y_true", "y_score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[警告] {PREDICTIONS_PATH} 缺少欄位 {missing}，沒有資料可畫，結束。")
        return None

    if df.empty or df["attack_raw"].isna().all():
        print(f"[警告] {PREDICTIONS_PATH} 沒有可用的 'attack_raw' 資料，沒有資料可畫，結束。")
        return None

    df["y_pred"] = (df["y_score"] > 0.5).astype(int)

    grouped = df.groupby("attack_raw").agg(
        rate=("y_pred", "mean"),
        count=("y_pred", "size"),
    )

    if OBSERVE_LABEL in grouped.index:
        others = grouped.drop(index=OBSERVE_LABEL).sort_values("rate", ascending=True)
        grouped = pd.concat([others, grouped.loc[[OBSERVE_LABEL]]])
    else:
        print(f"[警告] 資料裡沒有 '{OBSERVE_LABEL}' 這個類別，圖上不會有假陽性率那一條。")
        grouped = grouped.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(grouped))))
    y_positions = range(len(grouped))
    colors = [OBSERVE_COLOR if label == OBSERVE_LABEL else ATTACK_COLOR for label in grouped.index]

    ax.barh(y_positions, grouped["rate"], color=colors, edgecolor="black")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(grouped.index)
    ax.set_xlabel("Fraction Predicted as Attack (threshold = 0.5)")
    ax.set_xlim(0, 1.15)
    ax.set_title(
        "Per-Attack-Type Prediction Rate\n"
        "(red = observe / false positive rate, blue = attack type / recall)"
    )

    for y, rate, count in zip(y_positions, grouped["rate"], grouped["count"]):
        ax.text(rate + 0.01, y, f"{rate * 100:.1f}%  (n={count})", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "per_class_recall.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Info] 圖已存到 {out_path}")
    plt.close(fig)
    return grouped


if __name__ == "__main__":
    main()
