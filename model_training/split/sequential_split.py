from pathlib import Path
from typing import List, Tuple

from .base import SplitBase


class SequentialSplit(SplitBase):
    def split_to_chunks(self) -> List[Tuple[Path, int]]:
        self.distribute_chunk_size()

        records = self.it.records
        header = self.it.header_bytes

        with open(self.temp.temp_data_path, "rb") as infile:
            for chunk_file, chunk_size in self.chunk.chunk_path:
                chunk_records = records[:chunk_size]

                with open(chunk_file, "wb") as outfile:
                    outfile.write(header)
                    for rec in chunk_records:
                        infile.seek(rec.offset)
                        outfile.write(infile.read(rec.length))

                self.remove_record_by_index(0, chunk_size)
                records = self.it.records

        self.rewrite_to_temp()
        return self.chunk.chunk_path
