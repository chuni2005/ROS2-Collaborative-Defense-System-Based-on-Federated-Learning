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

    # def distribute_chunk_size():

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
        return self.chunk.chunk_path

    def _select_records(self, chunk_sizes: List[int]) -> List[List[RowRecord]]:
        records_by_label: Dict[str, List[RowRecord]] = defaultdict(list)
        for record in self.it.records:
            records_by_label[record.attack].append(record)

        for records in records_by_label.values():
            self.random.shuffle(records)

        chunks: List[List[RowRecord]] = [[] for _ in chunk_sizes]
        labels = sorted(records_by_label)
        if not labels:
            return chunks

        target_counts = self._target_counts(chunk_sizes, labels)
        for index, target in enumerate(target_counts):
            dominant = labels[index % len(labels)]
            self._take_records(
                chunks[index], records_by_label[dominant], target[dominant]
            )

        for index, target in enumerate(target_counts):
            for label, count in target.items():
                if label != labels[index % len(labels)]:
                    self._take_records(chunks[index], records_by_label[label], count)

        leftovers = [
            record
            for records in records_by_label.values()
            for record in records
        ]
        self.random.shuffle(leftovers)
        for index, chunk in enumerate(chunks):
            missing = chunk_sizes[index] - len(chunk)
            chunk.extend(leftovers[:missing])
            del leftovers[:missing]

        for chunk in chunks:
            self.random.shuffle(chunk)
        return chunks

    def _target_counts(
        self, chunk_sizes: List[int], labels: List[str]
    ) -> List[Dict[str, int]]:
        targets = []
        for index, chunk_size in enumerate(chunk_sizes):
            dominant = labels[index % len(labels)]
            dominant_count = min(chunk_size, int(chunk_size * 0.8 + 0.5))
            other_total = chunk_size - dominant_count
            others = [label for label in labels if label != dominant]
            counts = {label: 0 for label in labels}
            counts[dominant] = dominant_count

            if others:
                base, extra = divmod(other_total, len(others))
                for other_index, label in enumerate(others):
                    counts[label] = base + (other_index < extra)
            targets.append(counts)
        return targets

    @staticmethod
    def _take_records(
        chunk: List[RowRecord], available: List[RowRecord], count: int
    ) -> None:
        taken = min(count, len(available))
        chunk.extend(available[:taken])
        del available[:taken]
