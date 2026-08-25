from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, overload
import tempfile
import shutil
import csv
import os

from .structures import (
    RowRecord,
    IndexTable,
    Chunk,
    SourceData,
    TempData,
    OutputData,
    ChartImg,
)


class SplitBase(ABC):
    def __init__(
        self,
        src_path: str,
        tmp_dir: str,
        output_dir: str,
        chart_dir: str,
        chunk_size: int,
        chunk_num: int,
        hard_mode: bool = False,
    ):
        print("[Info] Initializing...")
        self.source = self._init_src(src_path)
        self.temp = self._init_tmp(tmp_dir)

        self._copy_source_to_temp()
        self.it = IndexTable(table_path=self.temp.temp_data_path)
        self.fresh_index_table()

        self.output = self._init_output(output_dir)
        self.chart = self._init_chart(chart_dir)
        self.chunk = self._init_chunk(chunk_num, chunk_size, hard_mode)
        print("[Info] Initialization complete.")

    def _init_src(self, src_path: str) -> SourceData:
        source = Path(src_path)
        if not source.is_file():
            raise FileNotFoundError(f"[Error] Source file `{src_path}` does not exist.")
        return SourceData(Path(src_path))

    def _init_tmp(self, tmp_dir: str) -> TempData:
        temp_dir = Path(tmp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        fd, unique_path = tempfile.mkstemp(
            prefix="data_", suffix=".csv", dir=str(temp_dir)
        )
        os.close(fd)
        return TempData(Path(unique_path))

    def _copy_source_to_temp(self) -> None:
        shutil.copy2(self.source.source_data_path, self.temp.temp_data_path)

    def _init_output(self, output_dir: str) -> OutputData:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return OutputData(output_dir, models_paths=[])

    def _init_chart(self, chart_dir: str) -> ChartImg:
        chart_dir = Path(chart_dir)
        chart_dir.mkdir(parents=True, exist_ok=True)
        return ChartImg(chart_dir, chart_paths=[])

    def _init_chunk(self, chunk_num: int, chunk_size: int, hard_mode: bool) -> Chunk:
        if chunk_num <= 0 or chunk_size <= 0:
            raise ValueError(
                "[Error] chunk_num and chunk_size must be positive integers."
            )

        total_needed_rows = chunk_num * chunk_size
        total_available_rows = self.it.row_num

        if hard_mode and (total_available_rows % total_needed_rows != 0):
            raise ValueError(
                f"[Error] Hard mode enabled: Total rows ({total_available_rows}) "
                f"must be perfectly divisible by requested total split size ({total_needed_rows})."
            )

        chunk_path = [
            (self.output.output_data_dir / Path(f"chunk_{i}.csv"), 0)
            for i in range(chunk_num)
        ]
        return Chunk(chunk_num=chunk_num, chunk_size=chunk_size, chunk_path=chunk_path)

    def fresh_index_table(self) -> None:
        self.it = IndexTable(table_path=self.temp.temp_data_path)

        with open(self.temp.temp_data_path, "rb") as f:
            header = f.readline()
            self.it.header_bytes = header

            header_text = header.decode("utf-8", errors="ignore").strip()
            headers = [
                h.strip().strip("\"'").lower() for h in next(csv.reader([header_text]))
            ]

            if "attack" not in headers:
                raise ValueError("[Error] Cannot find 'attack' label in header.")
            attack_col_idx = headers.index("attack")

            current_offset = f.tell()
            while line := f.readline():
                row_length = f.tell() - current_offset

                line_str = line.decode("utf-8", errors="ignore").strip()

                if line_str:
                    try:
                        columns = next(csv.reader([line_str]))
                    except csv.Error:
                        columns = line_str.split(",")

                    attack_label = (
                        columns[attack_col_idx].strip().strip("\"'")
                        if len(columns) > attack_col_idx
                        else "unknown"
                    )
                    self.it.records.append(
                        RowRecord(
                            offset=current_offset,
                            length=row_length,
                            attack=attack_label,
                        )
                    )
                current_offset = f.tell()

    @overload
    def remove_record_by_index(self, index: int) -> None: ...

    @overload
    def remove_record_by_index(self, index: int, count: int) -> None: ...

    def remove_record_by_index(self, index: int, count: Optional[int] = None) -> None:
        original_count = self.it.row_num

        if count is not None:
            if index < 0 or index > original_count:
                raise IndexError(
                    f"[Error] Index {index} out of range (amount: {original_count})."
                )
            if count < 0:
                raise IndexError("[Error] count must not be negative.")

            actual_count = min(count, original_count - index)
            end_index = index + actual_count

            del self.it.records[index:end_index]

        else:
            if index < 0 or index >= original_count:
                raise IndexError(
                    f"[Error] Index {index} out of range (amount: {original_count})."
                )
            self.it.records.pop(index)

    def rewrite_to_temp(self) -> Path:
        temp_file_path = self.temp.temp_data_path
        rewrite_target = temp_file_path.with_suffix(".csv.tmp")

        records = self.it.records
        header = self.it.header_bytes

        print("[Info] Creating new index structure...")

        with open(temp_file_path, "rb", buffering=1024 * 1024) as infile, open(
            rewrite_target, "wb", buffering=1024 * 1024
        ) as outfile:
            outfile.write(header)

            n = len(records)
            i = 0
            while i < n:
                run_start = records[i].offset
                run_len = records[i].length
                j = i + 1

                while j < n and records[j].offset == run_start + run_len:
                    run_len += records[j].length
                    j += 1

                infile.seek(run_start)
                outfile.write(infile.read(run_len))
                i = j

        os.replace(rewrite_target, temp_file_path)

        print("[Info] Rewrite Success.")
        print("[Info] Refreshing index table...")
        self.fresh_index_table()

        return temp_file_path

    def cleanup(self) -> None:
        try:
            self.temp.temp_data_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[Warn] Failed to clean up temp file: {exc}")

    @abstractmethod
    def split_to_chunks(self) -> List[Tuple[Path, int]]:
        pass

    def distribute_chunk_size(self) -> None:
        total_needed = self.chunk.chunk_size * self.chunk.chunk_num
        unit_size = self.chunk.chunk_size
        remain = 0

        if self.it.row_num < total_needed:
            unit_size = self.it.row_num // self.chunk.chunk_num
            remain = self.it.row_num % self.chunk.chunk_num

        self.chunk.chunk_path = [(path, unit_size) for path, _ in self.chunk.chunk_path]

        if remain and self.chunk.chunk_path:
            last_path, last_size = self.chunk.chunk_path[-1]
            self.chunk.chunk_path[-1] = (last_path, last_size + remain)
