from pathlib import Path
import random
from typing import List, Tuple

from .base import SplitBase
from .structures import RowRecord


class RandomSplit(SplitBase):
    def __init__(self, *args, random_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(random_seed)

    def split_to_chunks(self) -> List[Tuple[Path, int]]:
        self.distribute_chunk_size()
        records = self.it.records.copy()
        self.random.shuffle(records)

        selected_offsets = set()
        with open(self.temp.temp_data_path, "rb") as infile:
            start = 0
            for chunk_file, chunk_size in self.chunk.chunk_path:
                chunk_records = records[start : start + chunk_size]
                start += chunk_size

                with open(chunk_file, "wb") as outfile:
                    outfile.write(self.it.header_bytes)
                    for record in chunk_records:
                        infile.seek(record.offset)
                        outfile.write(infile.read(record.length))
                        selected_offsets.add(record.offset)

        self.it.records = [
            record for record in self.it.records if record.offset not in selected_offsets
        ]
        self.rewrite_to_temp()
        return self.chunk.chunk_path