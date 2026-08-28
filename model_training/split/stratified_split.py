from collections import Counter
from pathlib import Path
import random
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
import csv

from .base import SplitBase
from .structures import IndexTable, RowRecord


class StratifiedSplit(SplitBase):
    def __init__(self, *args, random_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(random_seed)

    def distribute_chunk_size(self):
        if self.it.row_num == 0:
            raise Exception(f"[Early Stop] index table is not enough！")

        attacks_in_chunk = [rec.attack for rec in self.it.records]
        attack_labels = set(attacks_in_chunk)
        attacks_count = len(attack_labels)

        # main_size = self.chunk.size // 10 * 8
        # in_other_size = self.chunk.size // 10 * 2  # total
        # every_chunk_size = main_size + in_other_size
        every_chunk_size = self.chunk.size
        
        for label in attack_labels:
            if every_chunk_size > attacks_in_chunk.count(label) or attacks_count<=1:
                raise Exception(f"[Early Stop] index table is not enough！")

        self.chunk.chunk_path = [
            (path, every_chunk_size) for path, _ in self.chunk.chunk_path
        ]
        print(f"[Info] set chunk size succesfull.")
    
    def split_to_chunks(self):
        self.distribute_chunk_size()

        record_pools = {}
        for rec in self.it.records:
            if rec.attack not in record_pools:
                record_pools[rec.attack] = []
            record_pools[rec.attack].append(rec)
        attack_labels = list(record_pools.keys())
        
        for i, (path, size) in enumerate(self.chunk.chunk_path):
            main_needed = (size // 10) * 8
            other_needed = (size // 10) * 2
            main_label = attack_labels[i % len(attack_labels)]
            
            chunk_records = []
            main_records = record_pools[main_label][:main_needed]
            chunk_records.extend(main_records)
            del record_pools[main_label][:main_needed]

            other_labels = [lbl for lbl in attack_labels if lbl != main_label]
            per_other_needed = other_needed // len(other_labels)

            for o_lbl in other_labels:
                other_records = record_pools[o_lbl][:per_other_needed]
                chunk_records.extend(other_records)
                del record_pools[o_lbl][:per_other_needed]

            new_chunk_table = IndexTable(
                table_path=path,
                header_bytes=self.it.header_bytes,
                records=chunk_records
            )
            
            # refresh index table records
            self.it.records = []
            for lbl in attack_labels:
                self.it.records.extend(record_pools[lbl])
            
            self._write_table_to_csv(path, new_chunk_table)
            print(f"[Info] Chunk {i} ({main_label} attack label) split finish in {path} with {len(chunk_records)} rows data.")

    def _write_table_to_csv(self, path: Path, table: IndexTable) -> None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            source_data_path = Path(self.temp.temp_data_path)

            with (
                open(source_data_path, "rb") as src_f,
                open(path, "wb") as tgt_f,
            ):
                tgt_f.write(self.it.header_bytes)

                for rec in table.records:
                    src_f.seek(rec.offset)
                    raw_line_bytes = src_f.read(rec.length)
                    tgt_f.write(raw_line_bytes)

            print(
                f"[Info] Success for turn table to {path}."
            )
            self.plot_attack_labels(path)

        except Exception as e:
            raise IOError(
                f"[Error] Failed to write index table to CSV at {path}. Reason: {str(e)}"
            )