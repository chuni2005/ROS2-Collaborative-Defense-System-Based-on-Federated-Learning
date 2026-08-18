import argparse
import os
import random
import shutil
import sys


def detect(args):
    if not os.path.exists(args.target_data):
        print("[Error] Source data not found.")
        sys.exit(1)

    if not os.path.exists(args.split_dir):
        os.makedirs(args.split_dir, exist_ok=True)

    if args.unit <= 0 or args.chunk <= 0 or args.unit % args.chunk != 0:
        print("[Error] SPLIT_UNIT must be a positive integer and divisible by CHUNK.")
        sys.exit(1)


def copy_data(data_path):
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
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple smaller CSV files based on the specified number of rows per file.")
    parser.add_argument("--target_data", type=str, help="Path to the input CSV file.")
    parser.add_argument("--split_dir", type=str, help="Directory where the split CSV files will be saved.")
    parser.add_argument("--unit", type=int, default=1000, help="Number of rows per process file. Default is 1000.")
    parser.add_argument("--chunk", type=int, default=1, help="Number of chunks to split the data for.")
    parser.add_argument("--random", action="store_true", help="Enable random sampling instead of chronological splitting.")
    args = parser.parse_args()

    detect(args)
    tmp = copy_data(args.target_data)
    if args.random:
        split_csv_random(tmp, args)
    else:
        split_csv_ordered(tmp, args)

if __name__ == "__main__":
    main()