import pandas as pd
import re
from pathlib import Path
import sys
from collections import defaultdict

def split_csv_by_attack_chunked(input_csv_path="test.csv", output_dir="attack_dataset_splits", chunk_size=100000):
    """
    使用分塊讀取 (Chunking) 的方式讀取大型 CSV 檔案，
    並根據分類欄位將資料分割成多個獨立的 CSV 檔案，避免記憶體不足 (OOM)。
    """
    input_path = Path(input_csv_path)
    out_dir_path = Path(output_dir)

    # 1. 檢查檔案是否存在
    if not input_path.exists():
        print(f"❌ 找不到檔案：{input_path.absolute()}")
        sys.exit(1)

    # 2. 建立存放分割檔案的資料夾
    out_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 分割後的檔案將儲存至：{out_dir_path.absolute()}")

    # 用來記錄各個標籤目前處理的總筆數
    label_counts = defaultdict(int)
    
    # 紀錄各個標籤的檔案是否是「第一次寫入」 (第一次寫入需要加上 Header)
    first_write = defaultdict(lambda: True)

    print(f"⏳ 正在使用分塊模式 (每次 {chunk_size} 筆) 讀取 {input_path.name}...")

    try:
        # 使用 chunksize 讀取大檔案
        chunks = pd.read_csv(input_path, low_memory=False, chunksize=chunk_size)
        
        target_col = None
        total_rows_processed = 0
        chunk_index = 1

        for df_chunk in chunks:
            # 在第一個 chunk 尋找目標欄位
            if target_col is None:
                possible_cols = ['attack', 'label', 'Label', 'class', 'Attack']
                for col in possible_cols:
                    if col in df_chunk.columns:
                        target_col = col
                        break
                
                if not target_col:
                    print(f"❌ 找不到可用來分類的標籤欄位！預期名稱：{possible_cols}")
                    sys.exit(1)
                print(f"🔍 成功鎖定分類欄位：'{target_col}'")

            # 取得這個區塊內所有的標籤
            unique_labels_in_chunk = df_chunk[target_col].dropna().unique()

            for label in unique_labels_in_chunk:
                # 篩選出該標籤的資料
                df_subset = df_chunk[df_chunk[target_col] == label]
                count_in_chunk = len(df_subset)
                
                if count_in_chunk == 0:
                    continue

                # 將標籤名稱轉換為安全的檔案名稱
                safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', str(label))
                safe_filename = re.sub(r'_+', '_', safe_filename).strip('_')
                save_path = out_dir_path / f"{safe_filename}.csv"
                
                # 判斷是否為第一次寫入此檔案
                is_first = first_write[label]
                
                # 以 'a' (附加) 模式寫入 CSV。如果是第一次寫入，才保留表頭 (header)
                df_subset.to_csv(save_path, mode='a', index=False, header=is_first)
                
                # 更新紀錄
                first_write[label] = False
                label_counts[label] += count_in_chunk
                
            total_rows_processed += len(df_chunk)
            print(f"  [區塊進度] 第 {chunk_index} 塊處理完成 (累計處理 {total_rows_processed} 筆資料)...")
            chunk_index += 1

    except Exception as e:
        print(f"❌ 處理過程中發生錯誤：{e}")
        sys.exit(1)

    # 結算總數
    print("-" * 45)
    print(f"✅ 全部分割完成！共掃描了 {total_rows_processed} 筆資料。")
    print("📊 各類型封包統計結果：")
    for label, count in label_counts.items():
        safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', str(label))
        safe_filename = re.sub(r'_+', '_', safe_filename).strip('_')
        print(f"  - [{label}]: {count} 筆 -> 儲存於 {safe_filename}.csv")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    
    # 你的大檔案路徑
    test_csv = base_dir / "test-001.csv"
    output_folder = base_dir / "attack_dataset_splits_001"
    
    # 執行分塊分割
    split_csv_by_attack_chunked(input_csv_path=test_csv, output_dir=output_folder)