from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(slots=True)
class RowRecord:
    offset: int
    length: int
    attack: str


@dataclass
class IndexTable:
    table_path: Path
    header_bytes: bytes = b""
    records: List[RowRecord] = field(default_factory=list)

    @property
    def row_num(self) -> int:
        return len(self.records)


@dataclass
class Chunk:
    chunk_path: List[Tuple[Path, int]] = field(default_factory=list)  # (path, rows)
    chunk_num: int = 0
    chunk_size: int = 0


@dataclass
class SourceData:
    source_data_path: Path


@dataclass
class TempData:
    temp_data_path: Path


@dataclass
class OutputData:
    output_data_dir: Path
    models_paths: List[str]


@dataclass
class ChartImg:
    chart_dir: Path
    chart_paths: List[str]