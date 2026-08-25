from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple

from .base import SplitBase
from .structures import RowRecord


class StratifiedSplit(SplitBase):
    def __init__(self, *args, random_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(random_seed)

    def split_to_chunks(self) -> List[Tuple[Path, int]]:
        self.distribute_chunk_size()
        chunk_sizes = [size for _, size in self.chunk.chunk_path]
        selected_by_chunk = self._select_records(chunk_sizes)
        selected_offsets = set()

        with open(self.temp.temp_data_path, "rb") as infile:
            for (chunk_file, chunk_size), records in zip(
                self.chunk.chunk_path, selected_by_chunk
            ):
                if len(records) != chunk_size:
                    raise RuntimeError(
                        "Stratified sampling produced an invalid chunk size."
                    )

                with open(chunk_file, "wb") as outfile:
                    outfile.write(self.it.header_bytes)
                    for record in records:
                        infile.seek(record.offset)
                        outfile.write(infile.read(record.length))
                        selected_offsets.add(record.offset)

        self.it.records = [
            record
            for record in self.it.records
            if record.offset not in selected_offsets
        ]
        self.rewrite_to_temp()
        return self.chunk.chunk_path

    def _select_records(self, chunk_sizes: List[int]) -> List[List[RowRecord]]:
        records_by_label: Dict[str, List[RowRecord]] = defaultdict(list)
        for record in self.it.records:
            records_by_label[record.attack].append(record)

        for records in records_by_label.values():
            self.random.shuffle(records)

        total_to_select = sum(chunk_sizes)
        label_counts = self._allocate_counts(
            total_to_select,
            {label: len(records) for label, records in records_by_label.items()},
        )
        selected_by_label = {
            label: records[:count]
            for label, records in records_by_label.items()
            for count in [label_counts[label]]
        }

        chunks: List[List[RowRecord]] = [[] for _ in chunk_sizes]
        for label, records in selected_by_label.items():
            for record in records:
                available_chunks = [
                    index
                    for index, chunk in enumerate(chunks)
                    if len(chunk) < chunk_sizes[index]
                ]
                target = self.random.choice(available_chunks)
                chunks[target].append(record)

        for chunk in chunks:
            self.random.shuffle(chunk)
        return chunks

    @staticmethod
    def _allocate_counts(total: int, available: Dict[str, int]) -> Dict[str, int]:
        if not available or total <= 0:
            return {label: 0 for label in available}

        total_available = sum(available.values())
        total = min(total, total_available)
        exact = {
            label: total * count / total_available for label, count in available.items()
        }
        allocation = {
            label: min(available[label], int(value)) for label, value in exact.items()
        }

        remaining = total - sum(allocation.values())
        labels = sorted(
            available,
            key=lambda label: exact[label] - allocation[label],
            reverse=True,
        )
        while remaining:
            changed = False
            for label in labels:
                if allocation[label] < available[label]:
                    allocation[label] += 1
                    remaining -= 1
                    changed = True
                    if not remaining:
                        break
            if not changed:
                break
        return allocation
