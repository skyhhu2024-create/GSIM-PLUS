import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import CLIMATE_FILE, GLOBAL_ATTR_FILE, STEP1_DIR, STEP2_DIR


def deduplicate_columns(columns):
    seen = {}
    result = []
    for col in columns:
        key = str(col).strip()
        count = seen.get(key, 0)
        if count == 0:
            result.append(key)
        else:
            result.append(f"{key}__dup{count}")
        seen[key] = count + 1
    return result


def read_csv_fallback(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1", "cp1252"]
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            df.columns = deduplicate_columns(df.columns)
            return df
        except Exception as exc:
            last_error = exc
    raise last_error


def load_station_lists():
    anchor_df = pd.read_csv(STEP1_DIR / "anchor_stations.csv")
    target_df = pd.read_csv(STEP1_DIR / "target_stations.csv")
    evaluable_df = pd.read_csv(STEP1_DIR / "evaluable_target_stations.csv")
    return anchor_df, target_df, evaluable_df


def load_static_features(station_ids):
    attr_df = read_csv_fallback(GLOBAL_ATTR_FILE)
    attr_df.columns = [str(c).strip() for c in attr_df.columns]
    if "gsim.no" in attr_df.columns:
        attr_df = attr_df.rename(columns={"gsim.no": "station_id"})

    climate_df = read_csv_fallback(CLIMATE_FILE)
    climate_df.columns = [str(c).strip() for c in climate_df.columns]
    if "gsim.no" in climate_df.columns and "station_id" not in climate_df.columns:
        climate_df = climate_df.rename(columns={"gsim.no": "station_id"})

    topo_cols = ["station_id", "latitude", "longitude", "altitude", "slope", "area_local"]
    soil_cols = ["snd", "slt", "scl"]
    climate_cols = ["station_id", "kg_code", "kg_major", "mean_flow_m3s"]

    attr_keep = [c for c in topo_cols + soil_cols if c in attr_df.columns]
    climate_keep = [c for c in climate_cols if c in climate_df.columns]

    static_df = attr_df[attr_keep].copy()
    climate_subset = climate_df[climate_keep].copy()
    overlap_cols = [c for c in climate_subset.columns if c in static_df.columns and c != "station_id"]
    if overlap_cols:
        climate_subset = climate_subset.drop(columns=overlap_cols)

    merged = static_df.merge(climate_subset, on="station_id", how="outer")

    hydro_path = STEP2_DIR / "station_hydro_features.csv"
    if hydro_path.exists():
        hydro_df = pd.read_csv(hydro_path)
        hydro_df["station_id"] = hydro_df["station_id"].astype(str)
        overlap_cols = [c for c in hydro_df.columns if c in merged.columns and c != "station_id"]
        if overlap_cols:
            merged = merged.drop(columns=overlap_cols)
        merged = merged.merge(hydro_df, on="station_id", how="left")

    merged = merged[merged["station_id"].isin(station_ids)].copy()
    return merged


def main():
    anchor_df, target_df, evaluable_df = load_station_lists()
    station_ids = set(anchor_df["station_id"]).union(set(target_df["station_id"]))

    print("Loading static features...")
    static_features = load_static_features(station_ids)
    static_features["category"] = static_features["station_id"].map(
        lambda x: "anchor" if x in set(anchor_df["station_id"]) else "target"
    )
    static_features["evaluable_target"] = static_features["station_id"].isin(set(evaluable_df["station_id"]))

    static_path = STEP2_DIR / "station_features.csv"
    static_features.to_csv(static_path, index=False)

    summary = {
        "n_features": int(len(static_features)),
        "n_anchor": int((static_features["category"] == "anchor").sum()),
        "n_target": int((static_features["category"] == "target").sum()),
        "n_evaluable_target": int(static_features["evaluable_target"].sum()),
        "static_only": True,
    }
    summary_path = STEP2_DIR / "feature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved static features to: {static_path}")


if __name__ == "__main__":
    main()
