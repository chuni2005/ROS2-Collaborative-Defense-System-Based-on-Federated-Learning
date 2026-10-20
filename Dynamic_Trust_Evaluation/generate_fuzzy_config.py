import pandas as pd
import numpy as np
import glob
import json

def main():
    excel_file = 'feature_list.xlsx'
    print(f"正在讀取特徵清單: {excel_file}...")
    df_features = pd.read_excel(excel_file, sheet_name='features_list')
    feature_names = df_features['feature_name'].tolist()

    csv_files = glob.glob('ROSPaCe_reduced/rospace_reduced_*.csv')
    print(f"找到 {len(csv_files)} 個 CSV 檔案，準備採用「分塊記憶體優化」讀取...\n")

    feature_stats = {feat: {'count': 0, 'sum': 0.0, 'sum_sq': 0.0} for feat in feature_names}
    label_col_candidates = ['label', 'Label', 'class', 'Attack', 'attack'] 
    normal_label = 'observe'
    total_normal_rows = 0

    # 從 0 號檔案取得正確的欄位名稱 (為了解決 1~14 號檔案沒有表頭的問題)
    header_file = 'ROSPaCe_reduced/rospace_reduced_0.csv'
    try:
        correct_columns = pd.read_csv(header_file, nrows=0).columns.tolist()
    except Exception as e:
        print(f"❌ 讀取 {header_file} 表頭失敗，程式終止: {e}")
        return

    # 計算平均值與變異數 (Streaming + 向量化加速)
    print("[階段一] 執行串流統計...")
    for i, file in enumerate(csv_files, 1):
        print(f"  - 正在處理第 {i}/{len(csv_files)} 個檔案: {file}")
        try:
            # 判斷檔案是否為 0 號，若是 1~14 號則強制套用 correct_columns 表頭
            if 'rospace_reduced_0.csv' in file:
                chunk_iter = pd.read_csv(file, chunksize=100000, low_memory=False)
            else:
                chunk_iter = pd.read_csv(file, names=correct_columns, chunksize=100000, low_memory=False, header=None)

            for chunk_idx, chunk in enumerate(chunk_iter):
                # 進度提示：每處理 50 萬筆印出一次狀態
                if chunk_idx % 5 == 0 and chunk_idx > 0:
                    print(f"     ...已處理該檔 {chunk_idx * 10} 萬筆資料")

                # 過濾正常流量
                current_label_col = next((col for col in label_col_candidates if col in chunk.columns), None)
                
                # 防呆：如果找不到標籤欄位，發出警告並跳過該區塊
                if not current_label_col:
                    print(f"     ⚠️ 找不到標籤欄位，跳過此區塊。")
                    continue
                
                if current_label_col:
                    chunk = chunk[chunk[current_label_col] == normal_label]
                
                total_normal_rows += len(chunk)

                # 向量化累積運算 (整批處理有效欄位)
                valid_cols = [c for c in feature_names if c in chunk.columns]
                if valid_cols:
                    numeric_chunk = chunk[valid_cols].apply(pd.to_numeric, errors='coerce')
                    chunk_counts = numeric_chunk.count()
                    chunk_sums = numeric_chunk.sum()
                    chunk_sq_sums = (numeric_chunk ** 2).sum()
                    
                    for feat in valid_cols:
                        if chunk_counts[feat] > 0:
                            feature_stats[feat]['count'] += chunk_counts[feat]
                            feature_stats[feat]['sum'] += chunk_sums[feat]
                            feature_stats[feat]['sum_sq'] += chunk_sq_sums[feat]
        except Exception as e:
            print(f"⚠️ 讀取 {file} 時發生錯誤: {e}")

    print(f"總共處理了 {total_normal_rows} 筆正常流量資料。")

    # 分類特徵
    normal_features = []
    long_tail_features = []
    fuzzy_config = {}

    for feat, stats in feature_stats.items():
        N = stats['count']
        if N > 1:
            mu = stats['sum'] / N
            var = (stats['sum_sq'] - (stats['sum']**2 / N)) / (N - 1)
            sigma = np.sqrt(max(var, 0))

            # 1. 剔除靜態特徵
            if sigma < 0.0001:
                continue
                
            # 2. 剔除類別型特徵
            if "type" in feat.lower() or "expert" in feat.lower():
                continue

            # 3. 分流：長尾特徵 vs 常態特徵
            if mu > 0 and (sigma / mu) > 5:
                long_tail_features.append(feat)
            else:
                normal_features.append((feat, mu, sigma))

    # 處理常態特徵 (使用 3-Sigma)
    for feat, mu, sigma in normal_features:
        fuzzy_config[feat] = {
            "method": "3-sigma",
            "mu": float(mu),
            "sigma": float(sigma),
            "L_vertices": [0.0, 0.0, float(mu + 2 * sigma)],
            "M_vertices": [float(mu + 1 * sigma), float(mu + 2 * sigma), float(mu + 3 * sigma)],
            "H_vertices": [float(mu + 2 * sigma), float(mu + 3 * sigma), float(mu + 10 * sigma)]
        }

    # 針對長尾特徵計算分位數
    if long_tail_features:
        print(f"\n[階段二] 發現 {len(long_tail_features)} 個長尾特徵，開始計算分位數 (Percentiles)...")
        # 建立字典存放長尾特徵的所有數據 (省記憶體只存特定欄位)
        long_tail_data = {feat: [] for feat in long_tail_features}

        for i, file in enumerate(csv_files, 1):
            print(f"  - 正在掃描第 {i}/{len(csv_files)} 個檔案: {file}")
            try:
                # 只讀取需要的欄位 (加上 label 欄位用來過濾)
                cols_to_use = long_tail_features + label_col_candidates
                
                # 判斷檔案是否為 0 號，若是 1~14 號則強制套用 correct_columns 表頭
                if 'rospace_reduced_0.csv' in file:
                    chunk_iter = pd.read_csv(file, usecols=lambda c: c in cols_to_use, chunksize=100000, low_memory=False)
                else:
                    chunk_iter = pd.read_csv(file, names=correct_columns, usecols=lambda c: c in cols_to_use, chunksize=100000, low_memory=False, header=None)
                
                for chunk in chunk_iter:
                    current_label_col = next((col for col in label_col_candidates if col in chunk.columns), None)
                    if current_label_col:
                        chunk = chunk[chunk[current_label_col] == normal_label]
                    
                    # 向量化轉換長尾特徵
                    valid_cols = [c for c in long_tail_features if c in chunk.columns]
                    if valid_cols:
                        numeric_chunk = chunk[valid_cols].apply(pd.to_numeric, errors='coerce')
                        for feat in valid_cols:
                            data = numeric_chunk[feat].dropna()
                            if len(data) > 0:
                                long_tail_data[feat].extend(data.astype(np.float32).tolist())
            except Exception as e:
                pass # 忽略沒有這些欄位的檔案

        # 計算分位數並生成頂點
        for feat, data_list in long_tail_data.items():
            if len(data_list) > 0:
                p75 = np.percentile(data_list, 75)
                p90 = np.percentile(data_list, 90)
                p95 = np.percentile(data_list, 95)
                p99 = np.percentile(data_list, 99)

                # 防呆：確保三角形不會變成 [0,0,0]
                # 強制讓 p75 < p90 < p95 < p99 保持微小的梯度
                if p90 <= p75: p90 = p75 + 0.1
                if p95 <= p90: p95 = p90 + 0.1
                if p99 <= p95: p99 = p95 + 0.5

                fuzzy_config[feat] = {
                    "method": "percentile",
                    "P75": float(p75),
                    "P90": float(p90),
                    "P95": float(p95),
                    "P99": float(p99),
                    "L_vertices": [0.0, 0.0, float(p90)],
                    "M_vertices": [float(p75), float(p90), float(p95)],
                    "H_vertices": [float(p90), float(p95), float(p99 * 5)] # H 的尾巴拉長
                }

    # 輸出結果
    output_file = 'fuzzy_threshold_config.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fuzzy_config, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 計算完成！")
    print(f"總共產出 {len(fuzzy_config)} 個特徵的模糊邊界設定。")
    print(f"(其中 {len(normal_features)} 個使用 3-Sigma，{len(long_tail_features)} 個使用分位數法)")
    print(f"結果已儲存至: {output_file}")

if __name__ == "__main__":
    main()