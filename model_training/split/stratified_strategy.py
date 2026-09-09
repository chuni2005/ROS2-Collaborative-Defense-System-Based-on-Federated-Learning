from pathlib import Path
from typing import List, Tuple

from .splitter import SplitStrategy, Splitter
from .structures import IndexTable, RowRecord


class StratifiedStrategy(SplitStrategy):
    def distribute_chunk_size(self, context: Splitter) -> None:
        if context.it.row_num == 0:
            raise Exception("[Early Stop] index table is not enough！")

        attacks_in_chunk = [rec.attack for rec in context.it.records]
        attack_labels = set(attacks_in_chunk)
        attacks_count = len(attack_labels)

        every_chunk_size = context.chunk.chunk_size

        for label in attack_labels:
            if every_chunk_size > attacks_in_chunk.count(label) or attacks_count <= 1:
                raise Exception("[Early Stop] index table is not enough！")

        context.chunk.chunk_path = [
            (path, every_chunk_size) for path, _ in context.chunk.chunk_path
        ]
        print("[Info] set chunk size successful.")

    def split_to_chunks(self, context: Splitter) -> None:
        context.distribute_chunk_size()

        record_pools: dict[str, List[RowRecord]] = {}
        for rec in context.it.records:
            record_pools.setdefault(rec.attack, []).append(rec)
        attack_labels = list(record_pools.keys())

        for i, (path, size) in enumerate(context.chunk.chunk_path):
            main_needed = (size // 10) * 8
            other_needed = (size // 10) * 2
            main_label = attack_labels[i % len(attack_labels)]

            chunk_it = IndexTable(header_bytes=context.it.header_bytes)
            main_records = record_pools[main_label][:main_needed]
            chunk_it.records.extend(main_records)
            del record_pools[main_label][:main_needed]

            other_labels = [lbl for lbl in attack_labels if lbl != main_label]
            per_other_needed = other_needed // len(other_labels) if other_labels else 0

            for o_lbl in other_labels:
                other_records = record_pools[o_lbl][:per_other_needed]
                chunk_it.records.extend(other_records)
                del record_pools[o_lbl][:per_other_needed]

            chunk_it.write_csv(context.temp.temp_data_path, path)
            print(
                f"[Info] Chunk {i} ({main_label} attack label) split finish "
                f"in {path} with {len(chunk_it.records)} rows data."
            )

        context.it.records = [rec for pool in record_pools.values() for rec in pool]
        return context.chunk.chunk_path