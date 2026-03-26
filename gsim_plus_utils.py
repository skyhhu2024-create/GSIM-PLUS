import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gsim_plus_config import LEGACY_SCRIPTS_DIR, MONTHLY_DIR, STUDY_END_YEAR, STUDY_START_YEAR


if str(LEGACY_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPTS_DIR))


def resolve_station_file(station_id):
    for suffix in (".mon", ".csv"):
        file_path = MONTHLY_DIR / f"{station_id}{suffix}"
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"Station file not found for {station_id}")


def read_station_series(station_id):
    file_path = resolve_station_file(station_id)
    df = pd.read_csv(file_path, comment="#")
    df.columns = [str(c).replace("\t", "").replace('"', "").replace("'", "").strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"No date column in {file_path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    if "MEAN" in df.columns:
        df["MEAN"] = pd.to_numeric(df["MEAN"], errors="coerce")
    if "n.available" in df.columns:
        df["n.available"] = pd.to_numeric(df["n.available"], errors="coerce")
    return df


def read_station_study_period(station_id):
    df = read_station_series(station_id)
    return df[(df["year"] >= STUDY_START_YEAR) & (df["year"] <= STUDY_END_YEAR)].copy()


def compute_valid_months(study_df):
    if "n.available" in study_df.columns:
        return int((study_df["n.available"] > 0).sum())
    if "MEAN" in study_df.columns:
        return int(study_df["MEAN"].notna().sum())
    return 0


def extract_gap_lengths(missing_mask):
    gap_lengths = []
    run = 0
    for flag in missing_mask:
        if flag:
            run += 1
        elif run > 0:
            gap_lengths.append(run)
            run = 0
    if run > 0:
        gap_lengths.append(run)
    return gap_lengths


def parse_gap_lengths(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [int(x) for x in text.split()]


def gap_bin(length):
    if length == 1:
        return "1"
    if length <= 3:
        return "2-3"
    if length <= 6:
        return "4-6"
    if length <= 12:
        return "7-12"
    if length <= 24:
        return "13-24"
    return "25+"


def weighted_choice_from_bins(rng, weights):
    bins = list(weights.keys())
    probs = np.array([weights[b] for b in bins], dtype=float)
    probs = probs / probs.sum()
    return bins[int(rng.choice(len(bins), p=probs))]
