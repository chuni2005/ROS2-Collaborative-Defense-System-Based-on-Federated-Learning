import argparse
import os
import random
import shutil
import sys


def detect(args):
    """檢查執行切分前的必要條件，任一項不成立就印出錯誤訊息並
    sys.exit(1)：來源檔案 args.target_data 必須存在；args.split_dir 不存在
    就自動建立；args.unit 與 args.chunk 都必須是正整數，且 unit 必須能被
    chunk 整除。
    """
    if not os.path.exists(args.target_data):
        print("[Error] Source data not found.")
        sys.exit(1)

    if not os.path.exists(args.split_dir):
        os.makedirs(args.split_dir, exist_ok=True)

    if args.unit <= 0 or args.chunk <= 0 or args.unit % args.chunk != 0:
        print("[Error] SPLIT_UNIT must be a positive integer and divisible by CHUNK.")
        sys.exit(1)


def copy_data(data_path):
    """把來源資料複製一份到 tmp/data.csv，之後的切分都對這份複本操作，
    不動原始檔案。如果 tmp/data.csv 已經存在（例如上次執行中斷留下的），
    直接沿用、不重新複製，並回傳這個複本的路徑。
    """
    tmp_file_path = os.path.join("tmp", "data.csv")

    if os.path.exists(tmp_file_path):
        print(f"[Info] {tmp_file_path} already exists. Skipping copy.")
    else:
        print("[Info] Copying source data to tmp folder...")

        if not os.path.exists(tmp_file_path):
            os.makedirs("tmp", exist_ok=True)
            tmp_file_path = os.path.join("tmp", "data.csv")
            shutil.copyfile(data_path, tmp_file_path)

    return tmp_file_path


def split_csv_ordered(tmp_file_path, args):
    """依照檔案原本的行序，從 tmp_file_path 開頭切出 args.unit 筆資料，
    平均分配進 args.chunk 個 chunk_N.csv，其餘資料寫回 tmp_file_path 供
    下一次呼叫繼續切。

    輸入：tmp_file_path — copy_data() 產生的複本路徑；args.unit 是這次要
    切出的筆數上限，args.chunk 是要分成幾個節點檔案，args.split_dir 是
    輸出目錄。
    輸出：無回傳值。副作用是在 args.split_dir 底下寫出 chunk_1.csv ~
    chunk_{args.chunk}.csv（各自帶表頭），並視情況覆寫或刪除
    tmp_file_path。

    怎麼做：逐行讀取檔案（不是整個載進記憶體），先讀出表頭，接著最多讀
    args.unit 行存進 lines；把這批行數平均分配給每個 chunk（除不盡時前
    extra 個 chunk 各多分一行）；分配完之後，把檔案剩下的內容（可能一行
    都沒有）串流寫進 remain_file_path；最後如果還有剩餘資料，用
    remain_file_path 取代 tmp_file_path（下次呼叫從這裡繼續切）；如果沒有
    剩餘，直接刪掉兩個檔案，代表來源資料已經切完。

    為什麼要串流處理：main.py 的 MainRunner 是設計來處理比記憶體大的來源
    檔案（一次只切 SPLIT_UNIT 行出來），如果這裡改成整個檔案讀進記憶體，
    就違背了這支腳本存在的目的。
    """
    print(f"[Info] Splitting source data into {args.unit} rows per file...")

    remain_file_path = tmp_file_path + ".remain"
    has_remaining = False

    with (
            open(tmp_file_path, "r", encoding="utf-8") as infile,
            open(remain_file_path, "w", encoding="utf-8", newline="") as f_remain,
        ):

        header = infile.readline()
        if not header:
            print("[Warning] Source file is empty!")
            return
        f_remain.write(header)

        lines = []
        for _ in range(args.unit):
            line = infile.readline()
            if not line:
                break
            lines.append(line)

        total_lines = len(lines)

        if total_lines == 0:
            print("[Warning] No data left to split.")
        else:
            if total_lines < args.chunk:
                print(
                    f"[Warning] Only {total_lines} row(s) remain this round, "
                    f"which is fewer than --chunk={args.chunk}. Some chunk "
                    f"files will still end up with 0 data rows (header only) "
                    f"because there simply isn't enough data left to give "
                    f"every chunk at least one row."
                )

            base = total_lines // args.chunk
            extra = total_lines % args.chunk
            sizes = [base + 1 if i < extra else base for i in range(args.chunk)]

            out_files = [
                open(
                    os.path.join(args.split_dir, f"chunk_{i}.csv"),
                    "w",
                    encoding="utf-8",
                    newline="",
                )
                for i in range(1, args.chunk + 1)
            ]

            try:
                for outfile in out_files:
                    outfile.write(header)

                idx = 0
                for outfile, size in zip(out_files, sizes):
                    for _ in range(size):
                        outfile.write(lines[idx])
                        idx += 1
            finally:
                for outfile in out_files:
                    outfile.close()

        while True:
            chunk_data = infile.read(1024**2)
            if not chunk_data:
                break
            has_remaining = True
            f_remain.write(chunk_data)

    if has_remaining:
        os.replace(remain_file_path, tmp_file_path)
        print("[Info] Successfully extracted chunks and updated original file.")
    else:
        os.remove(remain_file_path)
        os.remove(tmp_file_path)
        print("[Info] No remaining data to process.")

import csv
import os
import random

def split_csv_random(tmp_file_path, args):
    """跟 split_csv_ordered() 做同一件事（切出 args.unit 筆、分成
    args.chunk 份、其餘寫回 tmp_file_path），差別是用水塘抽樣（reservoir
    sampling）從整個檔案裡隨機取樣 args.unit 筆，而不是取檔案最前面的
    args.unit 行。

    原理（水塘抽樣）：維護一個大小固定為 args.unit 的樣本池
    sampled_rows。前 args.unit 筆資料直接放進池子；之後每讀到第 i 筆
    （i 從 args.unit 起算），以 args.unit/(i+1) 的機率（用
    random.randint(0, i) < args.unit 實作）跟池子裡隨機一筆互換，被換出來
    的那筆連同所有沒被選中的資料一起寫進 remain_file_path。這個做法能在
    只掃過一次檔案、不知道總筆數的情況下，保證每一筆資料被選進樣本池的
    機率相等。

    輸出檔案在寫入前先 random.shuffle(sampled_rows)，避免抽樣機制本身在
    池子裡留下的順序偏差被誤認成資料原本的順序。

    跟 split_csv_ordered() 的收尾邏輯不同：這裡用
    `os.path.getsize(remain_file_path) > 10`（而不是「有沒有資料」）判斷
    是否還有剩餘可處理，因為 remain_file_path 就算沒有資料列，也會因為
    表頭那一行永遠大於 0 bytes——用 10 這個閾值是為了跟「只剩表頭」的
    情況區分開來，不是嚴謹的資料量門檻。

    呼叫端要記得 args.seed 會影響這個函式的隨機性——main() 裡在呼叫這個
    函式之前呼叫 random.seed(args.seed)。
    """
    print(f"[Info] Safely sampling {args.unit} rows randomly using Python native CSV engine...")

    remain_file_path = tmp_file_path + ".remain"
    sampled_rows = []
    header = None

    with (
        open(tmp_file_path, "r", encoding="utf-8", newline="") as infile,
        open(remain_file_path, "w", encoding="utf-8", newline="") as f_remain,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(f_remain)

        header = next(reader, None)
        if not header:
            print("[Warning] Source file is empty!")
            return
        writer.writerow(header)

        for i, row in enumerate(reader):
            if i < args.unit:
                sampled_rows.append(row)
            else:
                j = random.randint(0, i)
                if j < args.unit:
                    writer.writerow(sampled_rows[j])
                    sampled_rows[j] = row
                else:
                    writer.writerow(row)

    total_lines = len(sampled_rows)

    if total_lines == 0:
        print("[Warning] No data left to split.")
        if os.path.exists(remain_file_path): os.remove(remain_file_path)
        if os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        return

    random.shuffle(sampled_rows)

    if total_lines < args.chunk:
        print(f"[Warning] Only {total_lines} row(s) sampled, fewer than --chunk={args.chunk}.")

    base = total_lines // args.chunk
    extra = total_lines % args.chunk
    sizes = [base + 1 if i < extra else base for i in range(args.chunk)]

    out_files = [
        open(os.path.join(args.split_dir, f"chunk_{i}.csv"), "w", encoding="utf-8", newline="")
        for i in range(1, args.chunk + 1)
    ]
    csv_writers = [csv.writer(f) for f in out_files]

    try:
        for cw in csv_writers:
            cw.writerow(header)

        idx = 0
        for cw, size in zip(csv_writers, sizes):
            for _ in range(size):
                if idx < len(sampled_rows):
                    cw.writerow(sampled_rows[idx])
                    idx += 1
    finally:
        for f in out_files:
            f.close()

    if os.path.exists(remain_file_path) and os.path.getsize(remain_file_path) > 10:
        os.replace(remain_file_path, tmp_file_path)
        print(f"[Info] Successfully extracted {total_lines} random rows and updated original file.")
    else:
        if os.path.exists(remain_file_path): os.remove(remain_file_path)
        if os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        print("[Info] All data has been consumed. No remaining data to process.")


def main():
    """指令列進入點：解析參數、呼叫 detect() 驗證前置條件，複製來源資料
    後依 --random 旗標決定呼叫 split_csv_random() 或 split_csv_ordered()。
    --random 搭配 --seed 時才會呼叫 random.seed()，讓隨機切分可重現；
    未加 --seed 則保留原本的非決定性行為。
    """
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple smaller CSV files based on the specified number of rows per file.")
    parser.add_argument("--target_data", type=str, help="Path to the input CSV file.")
    parser.add_argument("--split_dir", type=str, help="Directory where the split CSV files will be saved.")
    parser.add_argument("--unit", type=int, default=1000, help="Number of rows per process file. Default is 1000.")
    parser.add_argument("--chunk", type=int, default=1, help="Number of chunks to split the data for.")
    parser.add_argument("--random", action="store_true", help="Enable random sampling instead of chronological splitting.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for --random's reservoir sampling and shuffle, so the split is "
                              "reproducible across runs. Unset (default) keeps the original "
                              "non-deterministic behavior. Ignored in chronological mode.")
    args = parser.parse_args()

    detect(args)
    tmp = copy_data(args.target_data)
    if args.random:
        if args.seed is not None:
            random.seed(args.seed)
        split_csv_random(tmp, args)
    else:
        split_csv_ordered(tmp, args)

if __name__ == "__main__":
    main()