"""從 tmp/data.csv（split.py 水塘抽樣後未被抽中的剩餘資料）切出兩份互不重疊的
測試集：test_natural.csv（依原始類型比例，對照 chunk_6 整體 F1）與
test_rare.csv（稀有攻擊類型全數納入，用於逐類型 recall）。

test_natural 先對完整 remain pool 抽樣，test_rare 只能從 remain pool 扣掉
test_natural 已選走的資料列之後的池子裡抽——前一版做法相反（兩份的需求量
先加總、一次抽樣再切開分給兩邊），會讓 test_natural 的稀有類型配額被迫歸零，
因為 test_rare「全部收錄」的字面要求會把稀有類型僅有的樣本全部拿走。
"""

import csv
import random

SEED = 42

SRC_PATH = "tmp/data.csv"
OUT_DIR = "split_data"

NATURAL_TOTAL = 200_000
RARE_TOTAL = 200_000

RARE_TYPES = ("ros2 reflection", "ros2 node crashing")
SHARED_TYPES = (
    "nmap SYN flood",
    "observe",
    "nmap discovery",
    "ros2 reconnaissance",
    "metasploit SYN flood",
)
ALL_TYPES = SHARED_TYPES + RARE_TYPES


def count_by_attack_type(src_path):
    """掃過一次來源檔案，回傳 {攻擊類型: 筆數} 與表頭。用來算分層抽樣的
    比例權重，不保留任何一行的實際內容，記憶體只跟「類型數」成正比。
    """
    counts = {}
    with open(src_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        attack_idx = header.index("attack")
        for row in reader:
            key = row[attack_idx]
            counts[key] = counts.get(key, 0) + 1
    return header, counts


def largest_remainder_alloc(total, weights):
    """把整數 total 依 weights（{類型: 權重}）按比例分成整數配額，配額
    總和恰好等於 total。單純 `total * weight / sum(weights)` 取整數會因
    捨去誤差讓總和小於 total，所以先每個類型都取 floor，再把差額依小數
    部分大小由大到小逐一補 1 筆，直到補滿。
    """
    weight_sum = sum(weights.values())
    exact = {t: total * w / weight_sum for t, w in weights.items()}
    floors = {t: int(exact[t]) for t in weights}
    shortfall = total - sum(floors.values())
    by_fraction_desc = sorted(weights, key=lambda t: exact[t] - floors[t], reverse=True)
    for t in by_fraction_desc[:shortfall]:
        floors[t] += 1
    return floors


def sample_pass(src_path, attack_idx, need_by_type, full_collect_types, rng, skip_indices):
    """單次掃描來源檔案：need_by_type 指定的類型各自做水塘抽樣（逐類型獨立
    維護一個 reservoir），full_collect_types 指定的類型不抽樣、全數收集；
    skip_indices 給的行號一律跳過，讓呼叫端能把上次選中的行號餵進來，
    保證兩次呼叫的結果互不重疊。

    回傳 {類型: [(行號, 資料列), ...]}；保留行號是為了讓呼叫端可以把
    「這次選中的行號」再餵給下一次呼叫當 skip_indices。
    """
    reservoirs = {t: [] for t in need_by_type}
    seen_count = {t: 0 for t in need_by_type}
    collected = {t: [] for t in full_collect_types}

    with open(src_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for line_idx, row in enumerate(reader):
            if line_idx in skip_indices:
                continue
            attack_type = row[attack_idx]
            item = (line_idx, row)
            if attack_type in collected:
                collected[attack_type].append(item)
                continue
            if attack_type not in need_by_type:
                continue
            need = need_by_type[attack_type]
            i = seen_count[attack_type]
            pool = reservoirs[attack_type]
            if len(pool) < need:
                pool.append(item)
            else:
                j = rng.randint(0, i)
                if j < need:
                    pool[j] = item
            seen_count[attack_type] += 1

    reservoirs.update(collected)
    return reservoirs


def write_csv(path, header, rows):
    """把 rows 逐行寫成 CSV，第一行是 header。不做任何轉換或排序。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    """建立 test_natural.csv 與 test_rare.csv：第一階段對全部 7 種攻擊類型
    依 remain pool 原始比例分層抽樣出 test_natural 的 200,000 筆；第二階段
    把第一階段選中的行號當 skip_indices，從縮小後的池子抽 test_rare——
    稀有類型全數收錄，其餘五型依原始比例補滿。
    """
    rng = random.Random(SEED)

    header, counts = count_by_attack_type(SRC_PATH)
    attack_idx = header.index("attack")

    natural_alloc = largest_remainder_alloc(NATURAL_TOTAL, {t: counts[t] for t in ALL_TYPES})

    natural_reservoirs = sample_pass(
        SRC_PATH, attack_idx,
        need_by_type=natural_alloc, full_collect_types=(),
        rng=rng, skip_indices=set(),
    )
    for t in ALL_TYPES:
        assert len(natural_reservoirs[t]) == natural_alloc[t], (t, len(natural_reservoirs[t]), natural_alloc[t])

    natural_rows_with_idx = [item for t in ALL_TYPES for item in natural_reservoirs[t]]
    natural_used_indices = {idx for idx, _ in natural_rows_with_idx}

    remaining_after_natural = {t: counts[t] - natural_alloc[t] for t in RARE_TYPES}
    rare_remaining_budget = RARE_TOTAL - sum(remaining_after_natural.values())
    rare_shared_alloc = largest_remainder_alloc(rare_remaining_budget, {t: counts[t] for t in SHARED_TYPES})

    rare_reservoirs = sample_pass(
        SRC_PATH, attack_idx,
        need_by_type=rare_shared_alloc, full_collect_types=RARE_TYPES,
        rng=rng, skip_indices=natural_used_indices,
    )
    for t in SHARED_TYPES:
        assert len(rare_reservoirs[t]) == rare_shared_alloc[t], (t, len(rare_reservoirs[t]), rare_shared_alloc[t])
    for t in RARE_TYPES:
        assert len(rare_reservoirs[t]) == remaining_after_natural[t], (t, len(rare_reservoirs[t]), remaining_after_natural[t])

    natural_rows = [row for _, row in natural_rows_with_idx]
    rare_rows = [row for t in ALL_TYPES for _, row in rare_reservoirs[t]]

    rng.shuffle(natural_rows)
    rng.shuffle(rare_rows)

    write_csv(f"{OUT_DIR}/test_natural.csv", header, natural_rows)
    write_csv(f"{OUT_DIR}/test_rare.csv", header, rare_rows)

    print(f"seed={SEED}")
    print(f"test_natural.csv: {len(natural_rows)} rows")
    print(f"test_rare.csv: {len(rare_rows)} rows")
    for t in ALL_TYPES:
        print(f"  {t}: natural={natural_alloc[t]} rare={len(rare_reservoirs[t])} (pool={counts[t]})")


if __name__ == "__main__":
    main()
