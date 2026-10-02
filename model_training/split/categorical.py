import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_MAPPING_DIR = Path(__file__).resolve().parents[1] / "json"


def _mapping_path(column: str, mapping_dir: Path) -> Path:
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", column).rstrip(" .")
    return mapping_dir / f"{filename or 'column'}.json"


def _load_mapping(path: Path) -> tuple[str, Dict[str, int]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("categories"), dict):
        column = str(data.get("column", path.stem))
        entries = data["categories"]
    elif isinstance(data, dict):
        column = str(data.get("column", path.stem))
        entries = {key: value for key, value in data.items() if key != "column"}
    else:
        raise ValueError(f"Invalid category mapping JSON: {path}")

    mapping: Dict[str, int] = {}
    for value, entry in entries.items():
        category_id = entry.get("id") if isinstance(entry, dict) else entry
        if category_id is None:
            continue
        try:
            mapping[str(value)] = int(category_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid category ID for {value!r} in {path}"
            ) from error
    return column, mapping


def _save_mapping(
    path: Path, column: str, mapping: Dict[str, int], counts: Counter
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = {
        value: {"id": category_id, "count": counts.get(value, 0)}
        for value, category_id in mapping.items()
    }
    temporary_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as file:
            temporary_path = file.name
            json.dump(
                {"column": column, "categories": categories},
                file,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def encode_categorical_csv(
    csv_path: Path | str, mapping_dir: Path | str = DEFAULT_MAPPING_DIR
) -> Optional[Path]:
    """Encode nonnumeric columns in a CSV using persistent, shared category IDs."""
    csv_path = Path(csv_path)
    mapping_dir = Path(mapping_dir)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        headers = next(reader, None)
        if not headers:
            return None
        categorical = [False] * len(headers)
        for row in reader:
            for index, value in enumerate(row[: len(headers)]):
                if not value or categorical[index]:
                    continue
                try:
                    float(value)
                except ValueError:
                    categorical[index] = True

    categorical = [
        is_categorical and header.strip().lower() != "attack"
        for header, is_categorical in zip(headers, categorical)
    ]
    if not any(categorical):
        return None

    mapping_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    mappings: Dict[str, Dict[str, int]] = {}
    for header, is_categorical in zip(headers, categorical):
        if not is_categorical:
            continue
        path = _mapping_path(header, mapping_dir)
        if path in paths.values():
            raise ValueError(f"Category mapping filename collision for column {header!r}.")
        paths[header] = path
        if path.exists():
            saved_column, mappings[header] = _load_mapping(path)
            if saved_column != header:
                raise ValueError(
                    f"Category mapping file {path} belongs to column {saved_column!r}, "
                    f"not {header!r}."
                )
        else:
            mappings[header] = {}

    counts: Dict[str, Counter] = {header: Counter() for header in paths}
    next_ids = {
        header: max(mapping.values(), default=-1) + 1
        for header, mapping in mappings.items()
    }
    temporary_path: Optional[str] = None
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            output_headers = next(reader, [])
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=csv_path.parent,
                suffix=".encoded",
                delete=False,
            ) as target:
                temporary_path = target.name
                writer = csv.writer(target)
                writer.writerow(output_headers)
                for row in reader:
                    encoded_row = row.copy()
                    for index, header in enumerate(output_headers):
                        if header not in paths or index >= len(row):
                            continue
                        value = row[index]
                        if not value:
                            encoded_row[index] = "-1"
                            continue
                        mapping = mappings[header]
                        if value not in mapping:
                            mapping[value] = next_ids[header]
                            next_ids[header] += 1
                        counts[header][value] += 1
                        encoded_row[index] = str(mapping[value])
                    writer.writerow(encoded_row)

        for header, path in paths.items():
            _save_mapping(path, header, mappings[header], counts[header])
            print(
                f"[Splitter] Categorical column '{header}': "
                f"{len(mappings[header])} values mapped; "
                f"{sum(counts[header].values())} non-empty rows counted."
            )
        return Path(temporary_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def load_category_mappings(
    mapping_dir: Path | str = DEFAULT_MAPPING_DIR,
) -> Dict[str, Dict[str, int]]:
    mapping_dir = Path(mapping_dir)
    mappings: Dict[str, Dict[str, int]] = {}
    if not mapping_dir.exists():
        return mappings
    for path in mapping_dir.glob("*.json"):
        column, mapping = _load_mapping(path)
        mappings[column] = mapping
    return mappings


def apply_category_mappings(
    dataframe: pd.DataFrame, mapping_dir: Path | str = DEFAULT_MAPPING_DIR
) -> pd.DataFrame:
    """Apply saved IDs to raw string categories while retaining already encoded IDs."""
    for column, mapping in load_category_mappings(mapping_dir).items():
        if column not in dataframe.columns:
            continue
        known_ids = set(mapping.values())

        def encode(value: Any) -> Any:
            if pd.isna(value) or value == "":
                return -1
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                if float(value).is_integer() and int(value) in known_ids:
                    return value
            return mapping.get(str(value), -1)

        dataframe[column] = dataframe[column].map(encode)
    return dataframe