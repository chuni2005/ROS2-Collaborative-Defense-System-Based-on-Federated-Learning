from collections import Counter
import csv
from dataclasses import InitVar, dataclass, field
import os
from pathlib import Path
import stat
from typing import List, Optional, Tuple


@dataclass(slots=True)
class RowRecord:
    offset: int
    length: int
    attack: str


@dataclass
class IndexTable:
    table_path: Optional[Path] = None
    header_bytes: bytes = b""
    records: List[RowRecord] = field(default_factory=list)

    @property
    def row_num(self) -> int:
        return len(self.records)

    @property
    def label_distributed(self) -> dict:
        all_labels = [record.attack for record in self.records]
        return dict(Counter(all_labels))

    def load_csv(self, src_path: Optional[Path] = None) -> None:
        src_path = Path(src_path) if src_path is not None else self.table_path
        if src_path is None:
            raise ValueError("[Error] `src_path` (or `table_path`) is required.")
        self.table_path = src_path
        self.records.clear()

        with open(src_path, "rb") as f:
            header_bytes = f.readline()
            self.header_bytes = header_bytes

            header_text = header_bytes.decode("utf-8", errors="ignore").strip()
            headers = [
                h.strip().strip("\"'").lower() for h in next(csv.reader([header_text]))
            ]

            if "attack" not in headers:
                raise ValueError("[Error] Cannot find 'attack' label in header.")
            attack_col_idx = headers.index("attack")

            current_offset = f.tell()
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break

                row_length = len(line_bytes)

                while line_bytes.count(b'"') % 2 != 0:
                    next_line = f.readline()
                    if not next_line:
                        break
                    line_bytes += next_line
                    row_length += len(next_line)

                line_str = line_bytes.decode("utf-8", errors="ignore").strip()

                if line_str:
                    try:
                        columns = next(csv.reader([line_str]))

                        if len(columns) > attack_col_idx:
                            attack_label = columns[attack_col_idx].strip().strip("\"'")
                        else:
                            attack_label = "unknown"
                    except Exception:
                        attack_label = "unknown"

                    self.records.append(
                        RowRecord(
                            offset=current_offset,
                            length=row_length,
                            attack=attack_label,
                        )
                    )

                current_offset += row_length
        print(f"[IndexTable] Attack Label Distributed: {self.label_distributed}")

    def write_csv(self, src_path: Path, csv_path: Path) -> None:
        try:
            with (
                open(src_path, "rb") as src_f,
                open(csv_path, "wb") as tgt_f,
            ):
                tgt_f.write(self.header_bytes)

                for rec in self.records:
                    src_f.seek(rec.offset)
                    raw_line_bytes = src_f.read(rec.length)
                    tgt_f.write(raw_line_bytes)

            print(f"[IndexTable] Success for turn table to {csv_path}.")

        except Exception as e:
            raise IOError(
                f"[Error] Failed to write index table to CSV at {csv_path}. Reason: {str(e)}"
            )


@dataclass
class Chunk:
    chunk_data_dir: Path
    chunk_path: List[Tuple[Path, int]] = field(default_factory=list)  # (path, rows)
    chunk_num: int = 0
    chunk_size: int = 0

    def __post_init__(self):
        if self.chunk_num <= 0 or self.chunk_size <= 0:
            raise ValueError(
                "[Error] chunk_num and chunk_size must be positive integers."
            )

        self.chunk_data_dir = Path(self.chunk_data_dir)
        self.chunk_data_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_path = [
            (self.chunk_data_dir / Path(f"chunk_{i}.csv"), 0)
            for i in range(self.chunk_num)
        ]


@dataclass
class TempData:
    temp_data_dir: Path
    temp_data_path: Path = None
    src_path: InitVar[Path] = None

    def __post_init__(self, src_path: Path) -> None:
        if src_path is None:
            raise ValueError("[Error] `src_path` value needed.")

        src_path = Path(src_path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"[Error] Cannot find source file: {src_path}")

        self.temp_data_dir = Path(self.temp_data_dir).resolve()
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_data_path = self.temp_data_dir / f"link_{src_path.name}"

        if self.temp_data_path.exists():
            os.chmod(self.temp_data_path, stat.S_IWRITE)
            self.temp_data_path.unlink()

        # Hard link
        os.link(src_path, self.temp_data_path)
        os.chmod(self.temp_data_path, stat.S_IREAD)
        print(f"[TempData] Create Read-Only Link Finished: {self.temp_data_path}")

    def replace_with(self, new_content_path: Path) -> None:
        if self.temp_data_path.exists():
            os.chmod(self.temp_data_path, stat.S_IWRITE)
        os.replace(new_content_path, self.temp_data_path)
        os.chmod(self.temp_data_path, stat.S_IREAD)

    def __del__(self):
        try:
            path = getattr(self, "temp_data_path", None)
            if path and path.exists():
                os.chmod(path, stat.S_IWRITE)
                path.unlink()
        except Exception:
            pass


@dataclass
class OutputData:
    output_data_dir: Path
    models_paths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_data_dir = Path(self.output_data_dir)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ChartImg:
    chart_dir: Path
    chart_paths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chart_dir = Path(self.chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
