from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import math
import pandas as pd
import os
import random

from .structures import ChartImg, Chunk, IndexTable, OutputData, RowRecord, TempData


class SplitStrategy(ABC):
    @abstractmethod
    def distribute_chunk_size(self, context: "Splitter") -> None: ...

    @abstractmethod
    def split_to_chunks(self, context: "Splitter") -> None: ...


class Splitter:
    def __init__(
        self,
        src_path: str,
        tmp_dir: str,
        output_dir: str,
        chart_dir: str,
        chunk_size: int,
        chunk_num: int,
        strategy: SplitStrategy,
        hard_mode: bool = False,
    ):
        print("[Splitter] Initializing...")
        self.temp = TempData(Path(tmp_dir), src_path=Path(src_path))
        self.output = OutputData(Path(output_dir))
        self.chart = ChartImg(Path(chart_dir))
        self.chunk = Chunk(
            self.output.output_data_dir, chunk_num=chunk_num, chunk_size=chunk_size
        )
        self.hard_mode = hard_mode
        self.it = IndexTable(table_path=self.temp.temp_data_path)
        self.it.load_csv()
        self._validate_hard_mode()

        self.strategy = strategy
        print("[Splitter] Initialization complete.")

    def remove_record_by_index(self, index: int, count: int) -> None:
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

    def split_ratio_data(
        self,
        ratio: float,
        output_path: Path | str,
        random_seed: Optional[int] = None,
    ) -> Path:
        if not 0 < ratio < 1:
            raise ValueError("[Error] ratio must be between 0 and 1 (exclusive).")

        rng = random.Random(random_seed)
        records_by_label: dict[str, List[RowRecord]] = defaultdict(list)
        for record in self.it.records:
            records_by_label[record.attack].append(record)

        test_records: List[RowRecord] = []
        for label, records in records_by_label.items():
            pool = records.copy()
            rng.shuffle(pool)
            take = max(1, math.ceil(len(pool) * ratio))
            test_records.extend(pool[:take])
            print(f"[Splitter] Test split - {label}: {take}/{len(pool)} rows selected.")

        test_offsets = {record.offset for record in test_records}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        test_table = IndexTable(header_bytes=self.it.header_bytes, records=test_records)
        test_table.write_csv(self.temp.temp_data_path, output_path)

        self.it.records = [
            record for record in self.it.records if record.offset not in test_offsets
        ]

        print(
            f"[Splitter] Test dataset saved to {output_path} "
            f"({len(test_records)} rows); remaining rows: {self.it.row_num}."
        )
        return output_path

    # Strategy plumbing
    def set_strategy(self, strategy: SplitStrategy) -> None:
        self.strategy = strategy

    def distribute_chunk_size(self) -> None:
        self.strategy.distribute_chunk_size(self)

    def split_to_chunks(self) -> None:
        return self.strategy.split_to_chunks(self)

    def _validate_hard_mode(self) -> None:
        total_needed_rows = self.chunk.chunk_num * self.chunk.chunk_size
        if (
            self.hard_mode
            and total_needed_rows
            and self.it.row_num % total_needed_rows != 0
        ):
            raise ValueError(
                f"[Error] Hard mode enabled: Total rows ({self.it.row_num}) "
                f"must be perfectly divisible by requested total split size ({total_needed_rows})."
            )

    # Reporting
    def plot_attack_labels(self, csv_path: str, title: str) -> None:
        os.makedirs(self.chart.chart_dir, exist_ok=True)
        df = pd.read_csv(csv_path)

        if "attack" not in df.columns:
            raise ValueError("[Error] Cannot find attack column")

        label_counts = df["attack"].value_counts()

        plt.figure(figsize=(10, 6))
        label_counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Attack Label Distribution", fontsize=16)
        plt.xlabel("Attack Type", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = f"ALD_{title}.png"
        output_path = os.path.join(self.chart.chart_dir, filename)
        plt.savefig(output_path)
        plt.close()

        self.chart.chart_paths.append(output_path)
        print(f"[Splitter] Created chart successfully: {output_path}")
