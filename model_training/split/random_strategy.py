from pathlib import Path
import random
from typing import List, Tuple

from .splitter import SplitStrategy, Splitter


class RandomStrategy(SplitStrategy):
    def __init__(self, random_seed: int | None = None):
        self.random = random.Random(random_seed)

    def distribute_chunk_size(self, context: Splitter) -> None:
        context.default_distribute_chunk_size()

    def split_to_chunks(self, context: Splitter) -> List[Tuple[Path, int]]:
        context.distribute_chunk_size()
        records = context.it.records.copy()
        self.random.shuffle(records)

        selected_offsets = set()
        with open(context.temp.temp_data_path, "rb") as infile:
            start = 0
            for chunk_file, chunk_size in context.chunk.chunk_path:
                chunk_records = records[start : start + chunk_size]
                start += chunk_size

                with open(chunk_file, "wb") as outfile:
                    outfile.write(context.it.header_bytes)
                    for record in chunk_records:
                        infile.seek(record.offset)
                        outfile.write(infile.read(record.length))
                        selected_offsets.add(record.offset)

        context.it.records = [
            record
            for record in context.it.records
            if record.offset not in selected_offsets
        ]
        return context.chunk.chunk_path
