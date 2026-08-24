from pathlib import Path
from typing import List, Tuple, Dict
import random
from collections import defaultdict
from .base import SplitBase

class SequentialSplit(SplitBase):
    def split_to_chunks(self) -> List[Tuple[Path, int]]:
        results = []
        records = self.it.records
        header = self.it.header_bytes
        
        # with open(self.temp.temp_data_path, "rb") as infile:
        #     for c_idx in range(self.chunk.chunk_num):
        #         chunk_file = self.output.output_data_dir / f"seq_chunk_{c_idx}.csv"
        #         start_row = c_idx * self.chunk.chunk_size
        #         end_row = start_row / self.chunk.chunk_size + self.chunk.chunk_size

        #         chunk_records = records[start_row:end_row]

        #         with open(chunk_file, "wb") as outfile:
        #             outfile.write(header)
        #             for rec in chunk_records:
        #                 infile.seek(rec.offset)
        #                 outfile.write(infile.read(rec.length))

        #         results.append((chunk_file, len(chunk_records)))
        #         self.output.models_paths.append(str(chunk_file))

        # print(f"[Info]")
        # return results