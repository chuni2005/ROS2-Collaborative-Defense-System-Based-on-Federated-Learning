"""plot_f1.py / plot_recall.py 共用的小工具：讀 metrics_rounds.csv、指派可辨識的線條樣式、存圖。

抽成共用模組是因為這兩支腳本除了「要畫哪一欄」之外，其餘邏輯（讀檔、檢查欄位、
round 只有一種值時改畫長條圖、上色/線型/marker 分配、存檔）完全一樣，兩份幾乎
一樣的程式碼比一個小共用模組更容易漏改、改一邊忘了改另一邊。

不是給 plot_roc.py / plot_per_class_recall.py 用的——那兩支的資料來源、圖表
邏輯都不一樣，硬套這裡的函式反而繞遠路。
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
    pass  # 舊版 Python 或非一般 stream 沒有 reconfigure，忽略即可

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
METRICS_ROUNDS_CSV = RESULTS_DIR / "metrics_rounds.csv"

REQUIRED_COLUMNS = ["scenario", "round", "accuracy", "precision", "recall", "f1", "auc"]

# 黑白列印也要能分辨每條線/每根長條，所以顏色、線型、marker、長條網底各自獨立循環
# （四個清單長度不同，同一組合要循環超過 40 個 scenario 才會重複，一般情境數用不到）。
COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCHES = ["/", "\\", "x", ".", "o", "+", "*", "-"]


def style_for(i):
    """回傳第 i 個 scenario 該用的 (color, linestyle, marker, hatch)。"""
    return (
        COLORS[i % len(COLORS)],
        LINESTYLES[i % len(LINESTYLES)],
        MARKERS[i % len(MARKERS)],
        HATCHES[i % len(HATCHES)],
    )


def load_metrics_rounds(value_column):
    """讀 metrics_rounds.csv，檢查欄位齊全、value_column 不是全空。

    檔案不存在、缺欄位、value_column 全部是 NaN、或整份是空表，都印出警告後
    回傳 None（呼叫端看到 None 就直接結束，不要往下畫圖），不丟例外、不印
    traceback。
    """
    if not METRICS_ROUNDS_CSV.exists():
        print(f"[警告] 找不到 {METRICS_ROUNDS_CSV}，沒有資料可畫，結束。")
        return None

    df = pd.read_csv(METRICS_ROUNDS_CSV)

    if df.empty:
        print(f"[警告] {METRICS_ROUNDS_CSV} 沒有任何資料列，結束。")
        return None

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[警告] {METRICS_ROUNDS_CSV} 缺少欄位 {missing_cols}，沒有資料可畫，結束。")
        return None

    if df[value_column].isna().all():
        print(f"[警告] {METRICS_ROUNDS_CSV} 的 '{value_column}' 欄位全部是空值，沒有資料可畫，結束。")
        return None

    return df


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Info] 圖已存到 {out_path}")
    return out_path


def plot_metric_by_round(value_column, y_label, title, output_filename):
    """畫某個 metric 隨 round 變化的圖，回傳存檔路徑；沒資料可畫則回傳 None。

    全檔只出現一種 round 值時（例如整份只有 round=1），折線圖畫不出趨勢，
    改畫長條圖比較每個 scenario 在那一個 round 的數值；round 有兩種以上的值
    才畫折線圖。

    回傳值是給呼叫端（例如 notebook）判斷「這次到底有沒有真的產圖」用的——
    單看輸出檔案存不存在會誤判，因為上一次成功執行留下的舊圖檔還在，這次
    如果資料不見了、提早 return，檔案不會被覆蓋也不會被刪除。
    """
    df = load_metrics_rounds(value_column)
    if df is None:
        return None

    scenarios = list(df["scenario"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    single_round = df["round"].nunique() == 1

    if single_round:
        the_round = df["round"].iloc[0]
        for i, scenario in enumerate(scenarios):
            value = df.loc[df["scenario"] == scenario, value_column].iloc[0]
            color, _, _, hatch = style_for(i)
            ax.bar(i, value, color=color, hatch=hatch, edgecolor="black", label=scenario)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_xlabel("Scenario")
        ax.set_title(f"{title} (round={the_round}, single round only)")
    else:
        for i, scenario in enumerate(scenarios):
            sub = df[df["scenario"] == scenario].sort_values("round")
            color, linestyle, marker, _ = style_for(i)
            ax.plot(
                sub["round"], sub[value_column],
                color=color, linestyle=linestyle, marker=marker,
                label=scenario,
            )
        ax.set_xlabel("Round")
        ax.set_title(title)

    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = save_figure(fig, output_filename)
    plt.close(fig)
    return out_path
