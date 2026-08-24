import argparse
import os
import csv
import shutil
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


tmp_dir = "tmp"
img_dir = "img"
tmp_data_path = os.path.join(tmp_dir, "remaining_data.csv")
tmp_index_path = os.path.join(tmp_dir, "index_table.csv")

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def copy_data(data_path):
    if os.path.exists(tmp_data_path):
        print(f"[Info] {tmp_data_path} already exists. Skipping copy.")
    else:
        print("[Info] Copying source data to tmp folder...")
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copyfile(data_path, tmp_data_path)

    return tmp_data_path


def random_oversampling(min_stock=5000):
    print("[Info] Scanning label metadata for global oversampling defense...")

    with open(tmp_data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        label_idx = header.index("attack") if "attack" in header else len(header) - 1

        label_seeds = {}
        for row in reader:
            if row:
                lbl = row[label_idx]
                if lbl not in label_seeds:
                    label_seeds[lbl] = []
                if len(label_seeds[lbl]) < min_stock:
                    label_seeds[lbl].append(row)

    has_boosted = False
    with open(tmp_data_path, "a", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        for lbl, rows_pool in label_seeds.items():
            current_count = len(rows_pool)
            if current_count < min_stock:
                shortage = min_stock - current_count
                print(f"[Error] Class '{lbl}' count ({current_count}) < {min_stock}. Global Defense: Cloning {shortage} rows into storage...")
                has_boosted = True
                for _ in range(shortage):
                    random_row = random.choice(rows_pool)
                    writer.writerow(random_row)

    if has_boosted:
        print("[Info] Global oversampling defense completed successfully. Vault is now well-balanced.")
    else:
        print("[Info] All classes are already above the safety stock level. No action needed.")


def build_index_table():
    index_table = []

    with open(tmp_data_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
    header = next(csv.reader([header_line]))
    label_idx = header.index("attack") if "attack" in header else len(header) - 1

    print(f"[Info] Building Ultra-lightweight Index Table (using tuples) to {tmp_index_path}...")

    with open(tmp_data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        for current_row_idx, row in enumerate(reader):
            if row and len(row) > label_idx:
                index_table.append((current_row_idx, row[label_idx]))
            elif row:
                print(f"[Warning] Row {current_row_idx} has only {len(row)} columns, expected index {label_idx}. Skipping.")

    os.makedirs(tmp_dir, exist_ok=True)
    with open(tmp_index_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["index", "label"])
        writer.writerows(index_table)

    print(f"[Info] Index Table CSV created with {len(index_table)} records (Memory Optimized).")
    return index_table


def plot_distribution(csv_file_path, title_text="Data Label"):
    if not os.path.exists(csv_file_path):
        print(f"[Warning] Target CSV file '{csv_file_path}' not found. Skipping chart generation.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{title_text}_{timestamp}.png"
    os.makedirs(img_dir, exist_ok=True)
    output_path = os.path.join(img_dir, file_name)

    print(f"[Info] {title_text} Reading '{csv_file_path}' to plot distribution to {output_path}...")

    label_counts = {}
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print(f"[Warning] CSV file '{csv_file_path}' is empty!")
            return

        label_col_name = "attack" if "attack" in header else header[-1]
        label_idx = header.index(label_col_name)

        for row in reader:
            if row:
                lbl = row[label_idx]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

    if not label_counts:
        print("[Warning] No record found in CSV. Skipping chart generation.")
        return

    labels = sorted(list(label_counts.keys()))
    counts = [label_counts[lbl] for lbl in labels]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=counts, hue=labels, palette="viridis", legend=False)
    plt.title(f"{title_text}\nTotal Samples: {sum(counts)}")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[Info] Distribution chart successfully saved as '{file_name}' from CSV source.")


def split_csv_random(chunk_paths, tmp_file_path, args):
    print(f"[Info] Randomly sampling {args.unit} rows into {args.chunk} chunks...")

    remain_file_path = tmp_file_path + ".remain"
    sampled_rows = []
    header = None

    with (
        open(tmp_file_path, "r", encoding="utf-8", newline="") as infile,
        open(remain_file_path, "w", encoding="utf-8", newline="") as f_remain,
    ):
        reader = csv.reader(infile)
        remain_writer = csv.writer(f_remain)

        header = next(reader, None)
        if header is None:
            print("[Warning] Source file is empty!")
            return
        remain_writer.writerow(header)

        for current_row_idx, row in enumerate(reader):
            if current_row_idx < args.unit:
                sampled_rows.append(row)
            else:
                replace_idx = random.randint(0, current_row_idx)
                if replace_idx < args.unit:
                    remain_writer.writerow(sampled_rows[replace_idx])
                    sampled_rows[replace_idx] = row
                else:
                    remain_writer.writerow(row)

    total_lines = len(sampled_rows)

    if total_lines == 0:
        print("[Warning] No data left in tmp to split.")
        if os.path.exists(remain_file_path):
            os.remove(remain_file_path)
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        return

    if total_lines < args.chunk:
        print(
            f"[Warning] Only {total_lines} row(s) sampled this round, "
            f"fewer than --chunk={args.chunk}. Empty chunks may occur."
        )

    random.shuffle(sampled_rows)

    base = total_lines // args.chunk
    extra = total_lines % args.chunk
    sizes = [base + 1 if i < extra else base for i in range(args.chunk)]

    out_files = [open(path, "w", encoding="utf-8", newline="") for path in chunk_paths]
    writers = [csv.writer(f) for f in out_files]

    try:
        for writer in writers:
            writer.writerow(header)

        idx = 0
        for writer, size in zip(writers, sizes):
            for _ in range(size):
                if idx < total_lines:
                    writer.writerow(sampled_rows[idx])
                    idx += 1
    finally:
        for f in out_files:
            f.close()

    print(f"[Info] Successfully created {args.chunk} shuffled chunks from {total_lines} random rows.")

    with open(remain_file_path, "r", encoding="utf-8", newline="") as f:
        remain_reader = csv.reader(f)
        next(remain_reader, None)  # skip header
        has_remaining = next(remain_reader, None) is not None

    if has_remaining:
        os.replace(remain_file_path, tmp_file_path)
        print(f"[Info] tmp folder updated. Selected data removed from vault.")
    else:
        os.remove(remain_file_path)
        os.remove(tmp_file_path)
        print("[Info] All data has been entirely consumed. tmp files cleaned up.")


def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple smaller CSV files based on the specified number of rows per file.")
    parser.add_argument("--target_data", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--split_dir", type=str, required=True, help="Directory where the split CSV files will be saved.")
    parser.add_argument("--unit", type=int, required=True, help="Number of rows per process file. Default is 1000.")
    parser.add_argument("--chunk", type=int, required=True, help="Number of chunks to split the data for.")
    parser.add_argument("--mode", type=str, required=True, choices=["Direct", "Random", "DD", "SR"], help="Split the dataset with split mode.")
    parser.add_argument("--gen", type=str, required=True, choices=["OS", "SMOTE", "NONE"], help="Generate Data.")
    args = parser.parse_args()

    # gen data
    if args.gen == "OS":
        random_oversampling()

    detect(args)
    tmp_path = copy_data(args.target_data)

    chunk_paths = [
        os.path.join(args.split_dir, f"chunk_{i}.csv")
        for i in range(1, args.chunk + 1)
    ]

    build_index_table() # attack label
    plot_distribution(tmp_index_path, "Remaining Data Labels")

    # mode
    if args.mode == "Direct":
        split_csv_ordered(chunk_paths, tmp_path, args)

    elif args.mode == "Random":
        split_csv_random(chunk_paths, tmp_path, args)

    for chunk_path in chunk_paths:
        plot_distribution(chunk_path, f"plot_distribution")


if __name__ == "__main__":
    main()


## budget
# import csv
# target_file = "../ROSPaCe_complete/ROSPaCe_complete_noperiodicity.csv"  # 換成你的檔案路徑

# with open(target_file, "r", encoding="utf-8") as f:
#     header = next(csv.reader([f.readline()]))

# for idx, col_name in enumerate(header):
#     print(f"index [{idx}]: {col_name}")