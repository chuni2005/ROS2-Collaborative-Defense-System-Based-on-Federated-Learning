from pathlib import Path
from typing import List, Tuple

from .splitter import SplitStrategy, Splitter


class SequentialStrategy(SplitStrategy):
    def distribute_chunk_size(self, context: Splitter) -> None:
        context.default_distribute_chunk_size()

    def split_to_chunks(self, context: Splitter) -> List[Tuple[Path, int]]:
        context.distribute_chunk_size()
        header = context.it.header_bytes

        with open(context.temp.temp_data_path, "rb") as infile:
            for chunk_file, chunk_size in context.chunk.chunk_path:
                chunk_records = context.it.records[:chunk_size]

                with open(chunk_file, "wb") as outfile:
                    outfile.write(header)
                    for rec in chunk_records:
                        infile.seek(rec.offset)
                        outfile.write(infile.read(rec.length))

                context.remove_record_by_index(0, chunk_size)

        return context.chunk.chunk_path
