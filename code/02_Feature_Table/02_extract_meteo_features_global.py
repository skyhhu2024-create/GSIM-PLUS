import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import METEO_NC, STEP2_DIR


def manual_standardize(df, columns):
    out = df.copy()
    for col in columns:
        mean = out[col].mean()
        std = out[col].std()
        out[col] = 0.0 if std == 0 or pd.isna(std) else (out[col] - mean) / std
    return out


def extract_meteo_features(stations_df):
    import netCDF4 as nc

    ds = nc.Dataset(METEO_NC, "r")
    lats = np.asarray(ds.variables["latitude"][:])
    lons = np.asarray(ds.variables["longitude"][:])

    station_ids = stations_df["station_id"].tolist()
    latitudes = pd.to_numeric(stations_df["latitude"], errors="coerce").to_numpy()
    longitudes = pd.to_numeric(stations_df["longitude"], errors="coerce").to_numpy()
    valid_mask = np.isfinite(latitudes) & np.isfinite(longitudes)

    result = pd.DataFrame({"station_id": station_ids})
    for col in ["tp_mean", "tp_std", "tp_cv", "t2m_mean", "t2m_std", "e_mean", "e_std"]:
        result[col] = np.nan

    if not valid_mask.any():
        ds.close()
        return result

    valid_idx = np.where(valid_mask)[0]
    lat_valid = latitudes[valid_mask]
    lon_valid = longitudes[valid_mask]

    lat_indices = np.array([int(np.argmin(np.abs(lats - lat))) for lat in lat_valid], dtype=int)
    lon_indices = np.array(
        [
            int(np.argmin(np.minimum(np.abs(lons - lon), 360 - np.abs(lons - lon))))
            for lon in lon_valid
        ],
        dtype=int,
    )

    chunk_size = int(os.getenv("GSIM_METEO_CHUNK", "1000") or "1000")
    for start in range(0, len(valid_idx), chunk_size):
        end = min(start + chunk_size, len(valid_idx))
        chunk_rows = valid_idx[start:end]
        lat_chunk = lat_indices[start:end]
        lon_chunk = lon_indices[start:end]
        print(f"  Meteo processed {end}/{len(valid_idx)} stations...")

        tp_mean = np.empty(len(chunk_rows), dtype=np.float64)
        tp_std = np.empty(len(chunk_rows), dtype=np.float64)
        t2m_mean = np.empty(len(chunk_rows), dtype=np.float64)
        t2m_std = np.empty(len(chunk_rows), dtype=np.float64)
        e_mean = np.empty(len(chunk_rows), dtype=np.float64)
        e_std = np.empty(len(chunk_rows), dtype=np.float64)

        for lat_value in np.unique(lat_chunk):
            local_pos = np.where(lat_chunk == lat_value)[0]
            lon_sub = lon_chunk[local_pos]

            tp_block = np.asarray(ds.variables["tp"][:, int(lat_value), lon_sub], dtype=np.float64)
            t2m_block = np.asarray(ds.variables["t2m"][:, int(lat_value), lon_sub], dtype=np.float64)
            e_block = np.asarray(ds.variables["e"][:, int(lat_value), lon_sub], dtype=np.float64)

            if tp_block.ndim == 1:
                tp_block = tp_block[:, None]
                t2m_block = t2m_block[:, None]
                e_block = e_block[:, None]

            tp_block = tp_block.T * 1000.0
            t2m_block = t2m_block.T - 273.15
            e_block = e_block.T

            tp_mean[local_pos] = tp_block.mean(axis=1)
            tp_std[local_pos] = tp_block.std(axis=1)
            t2m_mean[local_pos] = t2m_block.mean(axis=1)
            t2m_std[local_pos] = t2m_block.std(axis=1)
            e_mean[local_pos] = e_block.mean(axis=1)
            e_std[local_pos] = e_block.std(axis=1)

        tp_cv = np.divide(tp_std, tp_mean, out=np.zeros_like(tp_std), where=tp_mean > 0)

        result.loc[chunk_rows, "tp_mean"] = np.round(tp_mean, 2)
        result.loc[chunk_rows, "tp_std"] = np.round(tp_std, 2)
        result.loc[chunk_rows, "tp_cv"] = np.round(tp_cv, 4)
        result.loc[chunk_rows, "t2m_mean"] = np.round(t2m_mean, 2)
        result.loc[chunk_rows, "t2m_std"] = np.round(t2m_std, 2)
        result.loc[chunk_rows, "e_mean"] = np.round(e_mean, 4)
        result.loc[chunk_rows, "e_std"] = np.round(e_std, 4)

    ds.close()
    return result


def main():
    static_path = STEP2_DIR / "station_features.csv"
    if not static_path.exists():
        raise FileNotFoundError(f"Static feature file not found: {static_path}")
    if not METEO_NC.exists():
        raise FileNotFoundError(f"NetCDF file not found: {METEO_NC}")

    stations = pd.read_csv(static_path)
    test_limit = int(os.getenv("GSIM_TEST_LIMIT", "0") or "0")
    meteo_input = stations[["station_id", "latitude", "longitude"]].copy()
    if test_limit > 0:
        meteo_input = meteo_input.head(test_limit).copy()
        print(f"Testing mode: extracting meteo features for first {len(meteo_input)} stations")

    meteo_df = extract_meteo_features(meteo_input)

    meteo_cols = ["tp_mean", "tp_std", "tp_cv", "t2m_mean", "t2m_std", "e_mean", "e_std"]
    stations_clean = stations.drop(columns=[c for c in meteo_cols if c in stations.columns], errors="ignore")
    merged = stations_clean.merge(meteo_df, on="station_id", how="left")

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged[numeric_cols] = merged[numeric_cols].apply(lambda s: s.fillna(s.median()))
    scaled = manual_standardize(merged, numeric_cols)

    if test_limit > 0:
        merged_path = STEP2_DIR / "station_features_with_meteo_test.csv"
        scaled_path = STEP2_DIR / "station_features_scaled_test.csv"
        summary_path = STEP2_DIR / "meteo_summary_test.json"
    else:
        merged_path = STEP2_DIR / "station_features_with_meteo.csv"
        scaled_path = STEP2_DIR / "station_features_scaled.csv"
        summary_path = STEP2_DIR / "meteo_summary.json"

    merged.to_csv(merged_path, index=False)
    scaled.to_csv(scaled_path, index=False)

    summary = {
        "n_features": int(len(merged)),
        "n_numeric_features": int(len(numeric_cols)),
        "test_limit": int(test_limit),
        "meteo_source": str(METEO_NC),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved merged features to: {merged_path}")
    print(f"Saved scaled features to: {scaled_path}")


if __name__ == "__main__":
    main()
