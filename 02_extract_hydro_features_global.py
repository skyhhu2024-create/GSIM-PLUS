import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import GLOBAL_ATTR_FILE, HYBAS_DIR, HYDRORIVERS_SHP, STEP2_DIR


TARGET_CRS = "EPSG:3857"
RADIUS_SEQUENCE = [0.03, 0.08, 0.20, 0.50]
SEARCH_AREA_PENALTY_M = 10000.0
AREA_TOLERANCE = 0.30


def deduplicate_columns(columns):
    seen = {}
    result = []
    for col in columns:
        key = str(col).strip()
        count = seen.get(key, 0)
        result.append(key if count == 0 else f"{key}__dup{count}")
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


def load_station_inputs():
    feature_path = STEP2_DIR / "station_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing station feature file: {feature_path}")

    stations = pd.read_csv(feature_path)
    stations["station_id"] = stations["station_id"].astype(str)
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"]).copy()

    attr_df = read_csv_fallback(GLOBAL_ATTR_FILE)
    attr_df.columns = [str(c).strip() for c in attr_df.columns]
    if "gsim.no" in attr_df.columns:
        attr_df = attr_df.rename(columns={"gsim.no": "station_id"})
    attr_df["station_id"] = attr_df["station_id"].astype(str)

    keep_cols = [c for c in ["station_id", "area_local", "river", "station", "country"] if c in attr_df.columns]
    attr_subset = attr_df[keep_cols].copy()
    if "area_local" in attr_subset.columns:
        attr_subset["area_local"] = pd.to_numeric(attr_subset["area_local"], errors="coerce")

    merged = stations.merge(attr_subset, on="station_id", how="left")
    geometry = gpd.points_from_xy(merged["longitude"], merged["latitude"], crs="EPSG:4326")
    return gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")


def basin_files():
    return sorted(HYBAS_DIR.glob("hybas_*_lev12_v1c.shp"))


def assign_basin_attributes(stations_gdf):
    result = stations_gdf.copy()
    result["hybas_source"] = pd.Series(dtype="object")
    result["hybas_id"] = np.nan
    result["area_local_hybas_km2"] = np.nan
    result["upstream_area_km2"] = np.nan

    unresolved = result.index.copy()
    for shp in basin_files():
        if len(unresolved) == 0:
            break

        continent_code = shp.name.split("_")[1]
        basins = gpd.read_file(shp, engine="pyogrio", columns=["HYBAS_ID", "SUB_AREA", "UP_AREA", "geometry"])
        bounds = basins.total_bounds
        candidate_idx = result.loc[unresolved].cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].index
        if len(candidate_idx) == 0:
            continue

        subset = result.loc[candidate_idx, ["station_id", "geometry"]].copy()
        joined = gpd.sjoin(subset, basins, how="left", predicate="within")
        joined = joined.dropna(subset=["HYBAS_ID"]).drop_duplicates(subset=["station_id"])
        if len(joined) == 0:
            continue

        joined["station_id"] = joined["station_id"].astype(str)
        hybas_map = joined.set_index("station_id")["HYBAS_ID"].astype("int64")
        sub_area_map = joined.set_index("station_id")["SUB_AREA"].astype(float)
        up_area_map = joined.set_index("station_id")["UP_AREA"].astype(float)
        hit_ids = hybas_map.index.tolist()

        mask = result["station_id"].isin(hit_ids)
        result.loc[mask, "hybas_source"] = continent_code
        result.loc[mask, "hybas_id"] = result.loc[mask, "station_id"].map(hybas_map)
        result.loc[mask, "area_local_hybas_km2"] = result.loc[mask, "station_id"].map(sub_area_map)
        result.loc[mask, "upstream_area_km2"] = result.loc[mask, "station_id"].map(up_area_map)

        unresolved = result[result["hybas_id"].isna()].index
        print(f"  Basin matched {shp.name}: {len(hit_ids)} stations, unresolved {len(unresolved)}")

    return result


def pick_best_river(point_gdf, target_hybas_id, target_upstream_area):
    point_3857 = point_gdf.to_crs(TARGET_CRS).geometry.iloc[0]

    for radius in RADIUS_SEQUENCE:
        lon = float(point_gdf.geometry.x.iloc[0])
        lat = float(point_gdf.geometry.y.iloc[0])
        bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
        rivers = gpd.read_file(
            HYDRORIVERS_SHP,
            engine="pyogrio",
            bbox=bbox,
            columns=[
                "HYRIV_ID",
                "MAIN_RIV",
                "DIST_DN_KM",
                "DIST_UP_KM",
                "CATCH_SKM",
                "UPLAND_SKM",
                "DIS_AV_CMS",
                "HYBAS_L12",
                "geometry",
            ],
        )
        if len(rivers) == 0:
            continue

        if pd.notna(target_hybas_id):
            hyriv_hybas = pd.to_numeric(rivers["HYBAS_L12"], errors="coerce")
            exact = rivers[hyriv_hybas == int(target_hybas_id)].copy()
            if len(exact) > 0:
                rivers = exact

        rivers_3857 = rivers.to_crs(TARGET_CRS)
        rivers_3857["snap_distance_m"] = rivers_3857.geometry.distance(point_3857)

        if pd.notna(target_upstream_area) and float(target_upstream_area) > 0:
            rivers_3857["area_diff_ratio"] = (
                np.abs(rivers_3857["UPLAND_SKM"].astype(float) - float(target_upstream_area))
                / float(target_upstream_area)
            )
        else:
            rivers_3857["area_diff_ratio"] = 0.0

        rivers_3857["score"] = rivers_3857["snap_distance_m"] + np.where(
            rivers_3857["area_diff_ratio"] > AREA_TOLERANCE,
            SEARCH_AREA_PENALTY_M,
            0.0,
        )
        rivers_3857 = rivers_3857.sort_values(["score", "snap_distance_m", "area_diff_ratio"])
        if len(rivers_3857) > 0:
            return rivers_3857.iloc[0]
    return None


def extract_hydro_features(stations_gdf):
    rows = []
    total = len(stations_gdf)
    for i, (_, row) in enumerate(stations_gdf.iterrows(), start=1):
        point = gpd.GeoDataFrame(
            [[row["station_id"], row.geometry]],
            columns=["station_id", "geometry"],
            crs="EPSG:4326",
        )
        best = pick_best_river(point, row["hybas_id"], row["upstream_area_km2"])
        record = {
            "station_id": row["station_id"],
            "area_local_raw_km2": row.get("area_local", np.nan),
            "area_local_hybas_km2": row.get("area_local_hybas_km2", np.nan),
            "upstream_area_km2": row.get("upstream_area_km2", np.nan),
            "hybas_id": row.get("hybas_id", np.nan),
            "hybas_source": row.get("hybas_source", np.nan),
            "mean_flow_m3s": np.nan,
            "snapped_upland_skm": np.nan,
            "snap_distance_m": np.nan,
            "area_diff_ratio": np.nan,
            "hyriv_id": np.nan,
            "main_riv": np.nan,
            "dist_dn_km": np.nan,
            "dist_up_km": np.nan,
            "river_status": "no_match",
        }
        if best is not None:
            record.update(
                {
                    "mean_flow_m3s": float(best["DIS_AV_CMS"]) if pd.notna(best["DIS_AV_CMS"]) else np.nan,
                    "snapped_upland_skm": float(best["UPLAND_SKM"]) if pd.notna(best["UPLAND_SKM"]) else np.nan,
                    "snap_distance_m": float(best["snap_distance_m"]),
                    "area_diff_ratio": float(best["area_diff_ratio"]),
                    "hyriv_id": int(best["HYRIV_ID"]) if pd.notna(best["HYRIV_ID"]) else np.nan,
                    "main_riv": int(best["MAIN_RIV"]) if pd.notna(best["MAIN_RIV"]) else np.nan,
                    "dist_dn_km": float(best["DIST_DN_KM"]) if pd.notna(best["DIST_DN_KM"]) else np.nan,
                    "dist_up_km": float(best["DIST_UP_KM"]) if pd.notna(best["DIST_UP_KM"]) else np.nan,
                    "river_status": "matched",
                }
            )
        rows.append(record)
        if i % 200 == 0 or i == total:
            print(f"  River snapped {i}/{total} stations...")
    return pd.DataFrame(rows)


def merge_hydro_into_feature_files(hydro_df):
    hydro_cols = [c for c in hydro_df.columns if c != "station_id"]
    for name in ["station_features.csv", "station_features_with_meteo.csv"]:
        path = STEP2_DIR / name
        if not path.exists():
            continue
        base = pd.read_csv(path)
        base["station_id"] = base["station_id"].astype(str)
        base = base.drop(columns=[c for c in hydro_cols if c in base.columns], errors="ignore")
        merged = base.merge(hydro_df, on="station_id", how="left")
        merged.to_csv(path, index=False)
        print(f"  Updated feature file: {path}")


def main():
    stations_gdf = load_station_inputs()
    test_limit = int(os.getenv("GSIM_HYDRO_TEST_LIMIT", "0") or "0")
    if test_limit > 0:
        stations_gdf = stations_gdf.head(test_limit).copy()
        print(f"Testing mode: first {len(stations_gdf)} stations")

    print("Assigning HydroBASINS attributes...")
    stations_gdf = assign_basin_attributes(stations_gdf)

    print("Snapping stations to HydroRIVERS...")
    hydro_df = extract_hydro_features(stations_gdf)

    output_csv = STEP2_DIR / ("station_hydro_features_test.csv" if test_limit > 0 else "station_hydro_features.csv")
    hydro_df.to_csv(output_csv, index=False)
    print(f"Saved hydro feature table to: {output_csv}")

    if test_limit == 0:
        merge_hydro_into_feature_files(hydro_df)

    summary = {
        "n_stations": int(len(hydro_df)),
        "n_basin_matched": int(hydro_df["hybas_id"].notna().sum()),
        "n_river_matched": int((hydro_df["river_status"] == "matched").sum()),
        "mean_snap_distance_m": float(hydro_df["snap_distance_m"].dropna().mean()) if hydro_df["snap_distance_m"].notna().any() else None,
    }
    summary_path = STEP2_DIR / ("hydro_summary_test.json" if test_limit > 0 else "hydro_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
