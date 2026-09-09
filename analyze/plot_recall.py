"""畫各 scenario 的 Recall 隨 round 變化的圖。

只讀 analyze/results/metrics_rounds.csv，不重新跑模型、不碰 test.csv。讀檔、
欄位檢查、round 只有一種值時改畫長條圖、上色與線型規則都在 _plot_common.py，
這支腳本只指定要畫哪一欄。
"""
from _plot_common import plot_metric_by_round


def main():
    return plot_metric_by_round(
        value_column="recall",
        y_label="Recall",
        title="Recall by Round",
        output_filename="recall_by_round.png",
    )


if __name__ == "__main__":
    main()
