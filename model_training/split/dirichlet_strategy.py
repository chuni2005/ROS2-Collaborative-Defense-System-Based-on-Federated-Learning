from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Tuple

from .splitter import SplitStrategy, Splitter
from .structures import RowRecord


class DirichletStrategy(SplitStrategy):
    def __init__(self, alpha: float = 1.0, random_seed: int | None = None):
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero.")
        self.alpha = alpha
        self.random = random.Random(random_seed)

    def distribute_chunk_size(self, context: Splitter) -> None:
        context.default_distribute_chunk_size()

    def split_to_chunks(self, context: Splitter) -> List[Tuple[Path, int]]:
        context.distribute_chunk_size()
        chunk_sizes = [size for _, size in context.chunk.chunk_path]
        records_by_label = self._group_records(context)
        selected_by_chunk = self._select_records(records_by_label, chunk_sizes)
        selected_offsets = set()

        with open(context.temp.temp_data_path, "rb") as infile:
            for (chunk_file, chunk_size), records in zip(
                context.chunk.chunk_path, selected_by_chunk
            ):
                if len(records) != chunk_size:
                    raise RuntimeError(
                        "Dirichlet sampling produced an invalid chunk size."
                    )

                with open(chunk_file, "wb") as outfile:
                    outfile.write(context.it.header_bytes)
                    for record in records:
                        infile.seek(record.offset)
                        outfile.write(infile.read(record.length))
                        selected_offsets.add(record.offset)

        context.it.records = [
            record
            for record in context.it.records
            if record.offset not in selected_offsets
        ]
        return context.chunk.chunk_path

    def _group_records(self, context: Splitter) -> Dict[str, List[RowRecord]]:
        records_by_label: Dict[str, List[RowRecord]] = defaultdict(list)
        for record in context.it.records:
            records_by_label[record.attack].append(record)
        for records in records_by_label.values():
            self.random.shuffle(records)
        return records_by_label

    def _select_records(
        self, records_by_label: Dict[str, List[RowRecord]], chunk_sizes: List[int]
    ) -> List[List[RowRecord]]:
        chunks: List[List[RowRecord]] = [[] for _ in chunk_sizes]
        remaining = chunk_sizes.copy()

        for records in records_by_label.values():
            amount = min(len(records), sum(remaining))
            weights = [self.random.gammavariate(self.alpha, 1.0) for _ in chunks]
            allocation = self._allocate(amount, weights, remaining)

            start = 0
            for index, count in enumerate(allocation):
                chunks[index].extend(records[start : start + count])
                start += count
                remaining[index] -= count

        if sum(remaining):
            leftovers = [
                record
                for records in records_by_label.values()
                for record in records
                if all(record not in chunk for chunk in chunks)
            ]
            for index, capacity in enumerate(remaining):
                chunks[index].extend(leftovers[:capacity])
                del leftovers[:capacity]

        for chunk in chunks:
            self.random.shuffle(chunk)
        return chunks

    @staticmethod
    def _allocate(amount: int, weights: List[float], capacity: List[int]) -> List[int]:
        total_weight = sum(weights)
        exact = [amount * weight / total_weight for weight in weights]
        allocation = [min(int(value), limit) for value, limit in zip(exact, capacity)]
        remaining = amount - sum(allocation)

        order = sorted(
            range(len(weights)),
            key=lambda index: exact[index] - allocation[index],
            reverse=True,
        )
        while remaining:
            changed = False
            for index in order:
                if allocation[index] < capacity[index]:
                    allocation[index] += 1
                    remaining -= 1
                    changed = True
                    if not remaining:
                        break
            if not changed:
                break
        return allocation
