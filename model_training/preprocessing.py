import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

ATTACK_MAPPING = {
    "observe": 0,
    "metasploit SYN flood": 1,
    "nmap discovery": 1,
    "nmap SYN flood": 1,
    "ros2 node crashing": 1,
    "ros2 reconnaissance": 1,
    "ros2 reflection": 1,
}

DEFAULT_CATEGORY_MAPS_FILENAME = "category_maps.json"


def convert_type(value):
    if (isinstance(value, (int, float, np.number)) and not pd.isna(value)) and not isinstance(
        value, bool
    ):
        return value

    if pd.isna(value) or value == "":
        return ""

    try:
        return pd.to_numeric(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


def _clean_for_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [col for col in df.columns if col != "attack"]

    if hasattr(df, "map"):
        df[feature_cols] = df[feature_cols].map(convert_type)
    else:
        df[feature_cols] = df[feature_cols].applymap(convert_type)

    df.replace([np.inf, -np.inf], -1, inplace=True)
    df.fillna(-1, inplace=True)
    df = df.dropna(thresh=1, axis=1)

    df = df.drop(
        columns=[col for col in df.columns if "Unnamed" in col or "timestamp" in col],
        errors="ignore",
    )
    return df


def build_category_maps(
    df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None
) -> Dict[str, List[str]]:
    cleaned = _clean_for_encoding(df)
    non_numeric_cols = cleaned.select_dtypes(exclude=[np.number, "bool"]).columns

    maps: Dict[str, List[str]] = {}
    for col in non_numeric_cols:
        if col == "attack":
            continue
        categories = pd.Categorical(cleaned[col]).categories
        maps[col] = [str(c) for c in categories]

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(maps, f, ensure_ascii=False, indent=2)
        print(f"[Preprocessing] Saved category maps ({len(maps)} columns) to {save_path}")

    return maps


def load_category_maps(path: Union[str, Path]) -> Dict[str, List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"category_maps file not found at {path!r}. Run the split/init step "
            "(MainRunner) once so it can be built from the full dataset before "
            "starting the server/clients."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_data(
    df: pd.DataFrame, category_maps: Dict[str, List[str]]
) -> pd.DataFrame:
    df = _clean_for_encoding(df)

    if "attack" in df.columns:
        df["attack"] = df["attack"].replace(ATTACK_MAPPING).infer_objects(copy=False)
        df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)

    non_numeric_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns
    for col in non_numeric_cols:
        if col == "attack":
            continue
        categories = category_maps.get(col)
        if categories is None:
            print(
                f"[Preprocessing][Warning] No fitted categories for column "
                f"{col!r}; falling back to a local (non-shared) encoding. "
                "Consider rebuilding category_maps.json from the full dataset."
            )
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = pd.Categorical(df[col], categories=categories).codes

    df.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in df.columns]
    return df.astype(np.float32)