import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import K_NEIGHBORS, STEP1_DIR, STEP2_DIR, STEP8_DIR, STUDY_END_YEAR, STUDY_START_YEAR
from gsim_plus_utils import read_station_study_period
from validation_wrappers import core, load_anchor_data, load_target_base_data, train_reusable_models


GROUP_WEIGHTS = {
    "hydro": 0.40,
    "climate": 0.25,
    "topo": 0.15,
    "spatial": 0.10,
    "soil": 0.10,
}

FEATURE_PRIORS = {
    "hydro": {
        "mean_flow_m3s": 0.50,
        "upstream_area_km2": 0.35,
        "area_local_hybas_km2": 0.15,
    },
    "climate": {
        "tp_mean": 0.24,
        "tp_std": 0.20,
        "tp_cv": 0.16,
        "t2m_mean": 0.16,
        "t2m_std": 0.12,
        "e_mean": 0.08,
        "e_std": 0.04,
    },
    "topo": {
        "altitude": 0.45,
        "slope": 0.55,
    },
    "spatial": {
        "latitude": 0.50,
        "longitude": 0.50,
    },
    "soil": {
        "snd": 0.34,
        "slt": 0.33,
        "scl": 0.33,
    },
}

HYDRO_LOG_COLS = {"mean_flow_m3s", "upstream_area_km2", "area_local_hybas_km2"}
LOW_FLOW_MEDIAN_THRESHOLD = 0.01


def cdist(XA, XB):
    diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def build_segments(indices):
    if len(indices) == 0:
        return []
    idx = sorted(int(i) for i in indices)
    segments = []
    start = idx[0]
    prev = idx[0]
    for value in idx[1:]:
        if value == prev + 1:
            prev = value
            continue
        segments.append((start, prev))
        start = value
        prev = value
    segments.append((start, prev))
    return segments


def choose_rule(seg_len):
    if seg_len <= 3:
        return "Q1"
    if seg_len <= 24:
        return "Q2"
    return "Q3"


def station_median_flow(base_data):
    values = np.asarray(base_data["y_original"], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def build_station_base_data(station_id):
    df = read_station_study_period(station_id)
    df["streamflow"] = pd.to_numeric(df["MEAN"], errors="coerce")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return {
        "dates": df["date"].values,
        "year": df["year"].values,
        "month": df["month"].values,
        "month_sin": df["month_sin"].values,
        "month_cos": df["month_cos"].values,
        "y_original": df["streamflow"].values,
        "valid": df["streamflow"].notna().values,
    }


def load_feature_inputs():
    feature_file = STEP2_DIR / "station_features_with_meteo.csv"
    if not feature_file.exists():
        feature_file = STEP2_DIR / "station_features.csv"
    features = pd.read_csv(feature_file)
    anchor_df = pd.read_csv(STEP1_DIR / "anchor_stations.csv")
    anchor_ids = set(anchor_df["station_id"])
    anchor_features = features[features["station_id"].isin(anchor_ids)].copy()
    return features, anchor_features


def get_feature_groups(features):
    groups = {
        "climate": ["tp_mean", "tp_std", "tp_cv", "t2m_mean", "t2m_std", "e_mean", "e_std"],
        "topo": ["altitude", "slope"],
        "soil": ["snd", "slt", "scl"],
        "spatial": ["latitude", "longitude"],
        "hydro": ["mean_flow_m3s", "area_local_hybas_km2", "upstream_area_km2"],
    }
    return {k: [f for f in v if f in features.columns] for k, v in groups.items()}


def learn_weights(features, feature_groups):
    final_weights = {}
    for group_name, feats in feature_groups.items():
        if not feats:
            continue
        g_weight = GROUP_WEIGHTS.get(group_name, 0.0)
        priors = FEATURE_PRIORS.get(group_name, {})
        feat_weights = {feat: float(priors.get(feat, 1.0)) for feat in feats}
        total = sum(feat_weights.values())
        for feat in feats:
            final_weights[feat] = g_weight * feat_weights[feat] / total

    total = sum(final_weights.values())
    return {k: v / total for k, v in final_weights.items()}


def build_anchor_to_anchor_topk():
    features, anchor_features = load_feature_inputs()
    feature_groups = get_feature_groups(features)
    weights = learn_weights(features, feature_groups)
    cols = list(weights.keys())

    anchor_numeric = anchor_features[cols].copy()
    for col in HYDRO_LOG_COLS:
        if col in anchor_numeric.columns:
            anchor_numeric[col] = np.log1p(pd.to_numeric(anchor_numeric[col], errors="coerce").clip(lower=0))

    scaler = StandardScaler()
    anchor_matrix = anchor_numeric.fillna(anchor_numeric.median(numeric_only=True)).fillna(0).values
    anchor_scaled = scaler.fit_transform(anchor_matrix)
    weight_vec = np.array([weights[c] for c in cols])
    weighted = anchor_scaled * weight_vec
    distances = cdist(weighted, weighted)

    anchor_ids = anchor_features["station_id"].values
    anchor_kg_major = anchor_features["kg_major"].fillna("").astype(str).values if "kg_major" in anchor_features.columns else np.array([""] * len(anchor_features))
    anchor_kg_code = anchor_features["kg_code"].fillna("").astype(str).values if "kg_code" in anchor_features.columns else np.array([""] * len(anchor_features))
    anchor_continent = anchor_features["hybas_source"].fillna("").astype(str).values if "hybas_source" in anchor_features.columns else np.array([""] * len(anchor_features))
    rows = []
    for i, target_id in enumerate(anchor_ids):
        target_major = anchor_kg_major[i]
        target_code = anchor_kg_code[i]
        target_continent = anchor_continent[i]
        same_continent = np.where(anchor_continent == target_continent)[0] if target_continent else np.array([], dtype=int)
        same_major = np.where(anchor_kg_major == target_major)[0] if target_major else np.array([], dtype=int)
        same_major = same_major[same_major != i]
        same_code_prefix = np.where(np.array([a[:1] for a in anchor_kg_code]) == target_code[:1])[0] if target_code else np.array([], dtype=int)
        same_code_prefix = same_code_prefix[same_code_prefix != i]
        same_continent = same_continent[same_continent != i]

        same_continent_major = np.intersect1d(same_continent, same_major) if len(same_continent) and len(same_major) else np.array([], dtype=int)
        same_continent_code = np.intersect1d(same_continent, same_code_prefix) if len(same_continent) and len(same_code_prefix) else np.array([], dtype=int)

        if len(same_continent_major) >= K_NEIGHBORS:
            candidate_idx = same_continent_major
            climate_filter = "continent+kg_major"
        elif len(same_continent_code) >= K_NEIGHBORS:
            candidate_idx = same_continent_code
            climate_filter = "continent+kg_code_prefix"
        elif len(same_continent) > 0:
            candidate_idx = same_continent
            climate_filter = "continent_only"
        elif len(same_major) >= K_NEIGHBORS:
            candidate_idx = same_major
            climate_filter = "kg_major"
        elif len(same_code_prefix) >= K_NEIGHBORS:
            candidate_idx = same_code_prefix
            climate_filter = "kg_code_prefix"
        else:
            candidate_idx = np.array([j for j in range(len(anchor_ids)) if j != i], dtype=int)
            climate_filter = "fallback_all"

        local_order = candidate_idx[np.argsort(distances[i, candidate_idx])[:K_NEIGHBORS]]
        for rank, j in enumerate(local_order, start=1):
            distance = float(distances[i, j])
            similarity = 1 / (1 + distance)
            rows.append(
                {
                    "target_station": target_id,
                    "anchor_station": anchor_ids[j],
                    "rank": rank,
                    "distance": round(distance, 4),
                    "similarity": round(float(similarity), 4),
                    "climate_filter": climate_filter,
                }
            )

    topk = pd.DataFrame(rows)
    weights_df = pd.DataFrame(list(weights.items()), columns=["feature", "weight"]).sort_values(
        "weight", ascending=False
    )
    return topk, weights_df


def recursive_predict_with_maml_product(model, val_data, device, calibration=None):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    with torch.no_grad():
        for idx in val_data["hide_indices"]:
            x_row = core.build_feature_row(val_data, std_series, idx)
            pred_std = float(model(torch.FloatTensor(x_row).unsqueeze(0).to(device)).cpu().numpy()[0, 0])
            pred_std = core.apply_linear_calibration(pred_std, calibration)
            pred_orig = float(core.from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
            if np.isfinite(pred_orig):
                predictions[(val_data["station_id"], int(idx))] = {"pred": pred_orig}
                std_series[idx] = pred_std
    return predictions


def method_maml_product(anchor_data, validation_set, similarity_df, trained_model, calibrated=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml_config = {
        "inner_lr": 0.05,
        "inner_steps": 10,
    }

    target_tasks = defaultdict(list)
    for _, row in similarity_df.iterrows():
        target_tasks[row["target_station"]].append({"anchor": row["anchor_station"], "similarity": row["similarity"]})

    predictions = {}
    for target_id, val_data in validation_set.items():
        if target_id not in target_tasks:
            continue

        anchor_support = core.build_anchor_support_tensors(anchor_data, target_tasks[target_id], device)
        self_support = core.build_self_support_tensors(val_data, device)
        adapted_model = core.adapt_maml_model(
            trained_model,
            anchor_support,
            self_support,
            inner_lr=maml_config["inner_lr"],
            inner_steps=maml_config["inner_steps"],
        )
        if adapted_model is None:
            continue
        calibration = core.fit_maml_station_calibration(adapted_model, val_data, device) if calibrated else None
        predictions.update(recursive_predict_with_maml_product(adapted_model, val_data, device, calibration))

    return predictions


def recursive_predict_with_donor_trend_product(model, anchor_data, target_task_list, val_data):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    hide_set = {int(i) for i in val_data["hide_indices"]}
    recursive_depth = 0
    for idx in val_data["hide_indices"]:
        idx = int(idx)
        if idx - 1 in hide_set:
            recursive_depth += 1
        else:
            recursive_depth = 0
        pred_std = core.predict_donor_trend_std(
            model,
            anchor_data,
            target_task_list,
            val_data,
            std_series,
            idx,
            recursive_depth=recursive_depth,
        )
        pred_orig = float(core.from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
        if np.isfinite(pred_orig):
            predictions[(val_data["station_id"], int(idx))] = {"pred": pred_orig}
            std_series[idx] = pred_std
    return predictions


def recursive_predict_with_maml_donor_trend_product(model, anchor_data, target_task_list, val_data):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    for idx in val_data["hide_indices"]:
        idx = int(idx)
        pred_std = core.predict_donor_trend_std_raw(
            model,
            anchor_data,
            target_task_list,
            val_data,
            std_series,
            idx,
        )
        pred_orig = float(core.from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
        if np.isfinite(pred_orig):
            predictions[(val_data["station_id"], int(idx))] = {"pred": pred_orig}
            std_series[idx] = pred_std
    return predictions


def method_donor_trend_product(anchor_data, validation_set, similarity_df):
    target_tasks = defaultdict(list)
    for _, row in similarity_df.iterrows():
        target_tasks[row["target_station"]].append({"anchor": row["anchor_station"], "similarity": row["similarity"]})

    predictions = {}
    for target_id, val_data in validation_set.items():
        task_list = target_tasks.get(target_id, [])
        if not task_list:
            continue
        model = core.fit_donor_trend_model(anchor_data, task_list, val_data)
        predictions.update(recursive_predict_with_donor_trend_product(model, anchor_data, task_list, val_data))
    return predictions


def method_maml_donor_trend_product(anchor_data, validation_set, similarity_df):
    target_tasks = defaultdict(list)
    for _, row in similarity_df.iterrows():
        target_tasks[row["target_station"]].append({"anchor": row["anchor_station"], "similarity": row["similarity"]})

    predictions = {}
    for target_id, val_data in validation_set.items():
        task_list = target_tasks.get(target_id, [])
        if not task_list:
            continue
        model = core.fit_donor_trend_model(anchor_data, task_list, val_data)
        predictions.update(recursive_predict_with_maml_donor_trend_product(model, anchor_data, task_list, val_data))
    return predictions


def method_product(anchor_data, validation_set, similarity_df, model_obj, method_name):
    if method_name in {"DonorTrend", "DonorTrend_Guarded"}:
        return method_donor_trend_product(anchor_data, validation_set, similarity_df)
    if method_name in {"MAML_DonorTrend", "MAML_DonorTrend_Guarded"}:
        return method_maml_donor_trend_product(anchor_data, validation_set, similarity_df)
    return method_maml_product(
        anchor_data,
        validation_set,
        similarity_df,
        trained_model=model_obj,
        calibrated=(method_name == "MAML_Calibrated"),
    )


def fill_anchor_station(station_id, anchor_data, similarity_df, model_obj, method_name):
    base_data = build_station_base_data(station_id)
    missing_indices = np.where(~base_data["valid"])[0]
    if len(missing_indices) == 0:
        rows = []
        for idx, date_val in enumerate(pd.to_datetime(base_data["dates"])):
            rows.append(
                {
                    "station_id": station_id,
                    "date": date_val.strftime("%Y-%m-%d"),
                    "year": int(base_data["year"][idx]),
                    "month": int(base_data["month"][idx]),
                    "observed_streamflow": float(base_data["y_original"][idx]) if not pd.isna(base_data["y_original"][idx]) else np.nan,
                    "final_streamflow": float(base_data["y_original"][idx]) if not pd.isna(base_data["y_original"][idx]) else np.nan,
                    "segment_length": 0,
                    "fill_method": "OBSERVED",
                    "quality_flag": "Q0",
                }
            )
        return pd.DataFrame(rows), {"station_id": station_id, "filled_points": 0, "status": "no_missing"}

    entry = core.build_validation_entry(station_id, base_data, missing_indices)
    if entry is None:
        return None, {"station_id": station_id, "filled_points": 0, "status": "insufficient_training"}

    effective_method = method_name
    if method_name == "MAML_DonorTrend_Guarded":
        median_flow = station_median_flow(base_data)
        if np.isfinite(median_flow) and median_flow < LOW_FLOW_MEDIAN_THRESHOLD:
            effective_method = "MAML"

    validation_set = {station_id: entry}
    pred_maml = method_product(anchor_data, validation_set, similarity_df, model_obj, effective_method)

    segments = build_segments(missing_indices)
    fill_map = {}
    filled_count = 0
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        quality_flag = choose_rule(seg_len)
        for idx in range(seg_start, seg_end + 1):
            key = (station_id, int(idx))
            pred_val = pred_maml.get(key, {}).get("pred")
            if pred_val is not None:
                filled_count += 1
            fill_map[int(idx)] = {
                "final_streamflow": float(pred_val) if pred_val is not None else np.nan,
                "segment_length": seg_len,
                "fill_method": effective_method if pred_val is not None else "UNFILLED",
                "quality_flag": quality_flag if pred_val is not None else "Q4",
            }

    summary = {
        "station_id": station_id,
        "filled_points": filled_count,
        "missing_points": int(len(missing_indices)),
        "status": "filled",
    }

    rows = []
    for idx, date_val in enumerate(pd.to_datetime(entry["dates"])):
        observed = entry["y_original_all"][idx]
        if pd.notna(observed):
            rows.append(
                {
                    "station_id": station_id,
                    "date": date_val.strftime("%Y-%m-%d"),
                    "year": int(entry["year"][idx]),
                    "month": int(entry["month"][idx]),
                    "observed_streamflow": float(observed),
                    "final_streamflow": float(observed),
                    "segment_length": 0,
                    "fill_method": "OBSERVED",
                    "quality_flag": "Q0",
                }
            )
        else:
            item = fill_map.get(
                int(idx),
                {"final_streamflow": np.nan, "segment_length": 0, "fill_method": "UNFILLED", "quality_flag": "Q4"},
            )
            rows.append(
                {
                    "station_id": station_id,
                    "date": date_val.strftime("%Y-%m-%d"),
                    "year": int(entry["year"][idx]),
                    "month": int(entry["month"][idx]),
                    "observed_streamflow": np.nan,
                    "final_streamflow": item["final_streamflow"],
                    "segment_length": item["segment_length"],
                    "fill_method": item["fill_method"],
                    "quality_flag": item["quality_flag"],
                }
            )
    return pd.DataFrame(rows), summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build GSIM-PLUS anchor product with a selected MAML variant.")
    parser.add_argument(
        "--method",
        choices=["MAML", "MAML_Calibrated", "MAML_DonorTrend", "MAML_DonorTrend_Guarded", "DonorTrend"],
        default="MAML",
        help="Method used for anchor-station filling.",
    )
    return parser.parse_args()


def anchor_output_dir(method_name):
    if method_name == "MAML":
        return STEP8_DIR
    return STEP8_DIR / method_name


def main():
    args = parse_args()
    anchor_ids = pd.read_csv(STEP1_DIR / "anchor_stations.csv")["station_id"].tolist()

    print("Building anchor-to-anchor similarity table...")
    anchor_similarity_df, weights_df = build_anchor_to_anchor_topk()
    anchor_similarity_path = STEP8_DIR / "anchor_to_anchor_top_5_similar_stations.csv"
    anchor_similarity_df.to_csv(anchor_similarity_path, index=False)
    weights_df.to_csv(STEP8_DIR / "anchor_to_anchor_feature_weights.csv", index=False)
    with open(STEP8_DIR / "anchor_to_anchor_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_anchor_targets": int(anchor_similarity_df["target_station"].nunique()),
                "n_rows": int(len(anchor_similarity_df)),
                "k_neighbors": K_NEIGHBORS,
                "mean_similarity": float(anchor_similarity_df["similarity"].mean()),
                "std_similarity": float(anchor_similarity_df["similarity"].std()),
            },
            f,
            indent=2,
        )

    print("Loading anchor data...")
    anchor_data = load_anchor_data(anchor_ids)
    anchor_task_data = load_target_base_data(anchor_ids)
    print("Training reusable models...")
    train_methods = ["MAML"] if args.method == "MAML_DonorTrend_Guarded" else [args.method]
    models = train_reusable_models(
        anchor_data,
        anchor_similarity_df,
        task_base_data=anchor_task_data,
        method_names=train_methods,
    )
    model_obj = models.get("maml")

    output_dir = anchor_output_dir(args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_fill_dir = output_dir / "GSIM_fill_anchor"
    anchor_fill_dir.mkdir(parents=True, exist_ok=True)

    full_series_path = output_dir / "gsim_plus_anchor_1995_2015_full_series.csv"
    infilled_points_path = output_dir / "gsim_plus_anchor_1995_2015_infilled_points.csv"

    if full_series_path.exists():
        full_series_path.unlink()
    if infilled_points_path.exists():
        infilled_points_path.unlink()

    full_written = False
    infilled_written = False
    total_filled_points = 0
    summaries = []

    for i, station_id in enumerate(anchor_ids, start=1):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(anchor_ids)} anchor stations...")
        try:
            station_df, summary = fill_anchor_station(station_id, anchor_data, anchor_similarity_df, model_obj, args.method)
            summaries.append(summary)
            if station_df is not None:
                station_df.to_csv(anchor_fill_dir / f"{station_id}.csv", index=False)
                station_df.to_csv(full_series_path, mode="a", header=not full_written, index=False)
                full_written = True

                infilled_df = station_df[station_df["quality_flag"] != "Q0"].copy()
                if len(infilled_df) > 0:
                    total_filled_points += int(infilled_df["final_streamflow"].notna().sum())
                    infilled_df.to_csv(
                        infilled_points_path,
                        mode="a",
                        header=not infilled_written,
                        index=False,
                    )
                    infilled_written = True
        except Exception:
            summaries.append({"station_id": station_id, "filled_points": 0, "status": "error"})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "gsim_plus_anchor_fill_summary.csv", index=False)

    product_summary = {
        "n_anchor_stations": int(len(anchor_ids)),
        "n_processed_stations": int(len(summary_df)),
        "n_filled_points": int(total_filled_points),
        "study_period": f"{STUDY_START_YEAR}-{STUDY_END_YEAR}",
        "production_method": args.method,
        "station_split_dir": str(anchor_fill_dir),
        "similarity_file": str(anchor_similarity_path),
    }
    with open(output_dir / "gsim_plus_anchor_product_summary.json", "w", encoding="utf-8") as f:
        json.dump(product_summary, f, indent=2)

    print(json.dumps(product_summary, indent=2))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
