import argparse
import json
import sys
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import STEP1_DIR, STEP3_DIR, STEP8_DIR, STUDY_END_YEAR, STUDY_START_YEAR
from gsim_plus_utils import read_station_study_period
from validation_wrappers import core, load_anchor_data, train_reusable_models


HYBRID_SCHEMES = {
    "maml_only": {
        "Q1": "MAML",
        "Q2": "MAML",
        "Q3": "MAML",
    },
    "maml_donor_trend_only": {
        "Q1": "MAML_DonorTrend",
        "Q2": "MAML_DonorTrend",
        "Q3": "MAML_DonorTrend",
    },
    "maml_donor_trend_guarded": {
        "Q1": "MAML_DonorTrend",
        "Q2": "MAML_DonorTrend",
        "Q3": "MAML_DonorTrend",
    },
    "donor_trend_only": {
        "Q1": "DonorTrend",
        "Q2": "DonorTrend",
        "Q3": "DonorTrend",
    },
    "maml_calibrated_only": {
        "Q1": "MAML_Calibrated",
        "Q2": "MAML_Calibrated",
        "Q3": "MAML_Calibrated",
    },
    "hybrid_v1": {
        "Q1": "RandomForest",
        "Q2": "Linear",
        "Q3": "MAML",
    },
    "hybrid_v2": {
        "Q1": "Linear",
        "Q2": "MAML",
        "Q3": "MAML",
    },
}
SKLEARN_MODEL_TYPES = {
    "RandomForest": "rf",
    "Linear": "linear",
    "KNN": "knn",
}
LOW_FLOW_MEDIAN_THRESHOLD = 0.01


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


def build_target_base_data(station_id):
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


def recursive_predict_with_sklearn_product(model, val_data):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    for idx in val_data["hide_indices"]:
        x_row = core.build_feature_row(val_data, std_series, idx).reshape(1, -1)
        pred_std = float(model.predict(x_row)[0])
        pred_orig = float(core.from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
        if np.isfinite(pred_orig):
            predictions[(val_data["station_id"], int(idx))] = {"pred": pred_orig}
            std_series[idx] = pred_std
    return predictions


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


def method_ml_product(validation_set, trained_model):
    predictions = {}
    for val_data in validation_set.values():
        predictions.update(recursive_predict_with_sklearn_product(trained_model, val_data))
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


def get_scheme_predictions(station_id, entry, anchor_data, similarity_df, models, scheme_rules):
    validation_set = {station_id: entry}
    required_methods = sorted(set(scheme_rules.values()))
    predictions_by_method = {}

    for method_name in required_methods:
        if method_name == "MAML_DonorTrend":
            predictions_by_method[method_name] = method_maml_donor_trend_product(anchor_data, validation_set, similarity_df)
            continue
        if method_name == "DonorTrend":
            predictions_by_method[method_name] = method_donor_trend_product(anchor_data, validation_set, similarity_df)
            continue

        if method_name in {"MAML", "MAML_Calibrated"}:
            predictions_by_method[method_name] = method_maml_product(
                anchor_data,
                validation_set,
                similarity_df,
                trained_model=models["maml"],
                calibrated=(method_name == "MAML_Calibrated"),
            )
            continue

        model_type = SKLEARN_MODEL_TYPES.get(method_name)
        if model_type is None:
            raise ValueError(f"Unsupported product method: {method_name}")
        predictions_by_method[method_name] = method_ml_product(
            validation_set,
            trained_model=models[model_type],
        )

    return predictions_by_method


def apply_low_flow_guard(scheme_rules, base_data, guard_low_flow=False):
    if not guard_low_flow:
        return dict(scheme_rules)
    median_flow = station_median_flow(base_data)
    if np.isfinite(median_flow) and median_flow < LOW_FLOW_MEDIAN_THRESHOLD:
        return {flag: ("MAML" if method == "MAML_DonorTrend" else method) for flag, method in scheme_rules.items()}
    return dict(scheme_rules)


def fill_station(station_id, anchor_data, similarity_df, models, scheme_rules, guard_low_flow=False):
    base_data = build_target_base_data(station_id)
    missing_indices = np.where(~base_data["valid"])[0]
    if len(missing_indices) == 0:
        full_rows = []
        for idx, date_val in enumerate(pd.to_datetime(base_data["dates"])):
            full_rows.append(
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
        return pd.DataFrame(full_rows), {"station_id": station_id, "filled_points": 0, "status": "no_missing"}

    entry = core.build_validation_entry(station_id, base_data, missing_indices)
    if entry is None:
        return None, {"station_id": station_id, "filled_points": 0, "status": "insufficient_training"}

    effective_scheme_rules = apply_low_flow_guard(scheme_rules, base_data, guard_low_flow=guard_low_flow)
    predictions_by_method = get_scheme_predictions(
        station_id,
        entry,
        anchor_data,
        similarity_df,
        models,
        effective_scheme_rules,
    )

    segments = build_segments(missing_indices)
    fill_map = {}
    filled_count = 0
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        quality_flag = choose_rule(seg_len)
        selected_method = effective_scheme_rules[quality_flag]
        selected_predictions = predictions_by_method.get(selected_method, {})
        for idx in range(seg_start, seg_end + 1):
            key = (station_id, int(idx))
            pred_val = selected_predictions.get(key, {}).get("pred")
            if pred_val is not None:
                filled_count += 1

            fill_map[int(idx)] = {
                "final_streamflow": float(pred_val) if pred_val is not None else np.nan,
                "segment_length": seg_len,
                "fill_method": selected_method if pred_val is not None else "UNFILLED",
                "quality_flag": quality_flag if pred_val is not None else "Q4",
            }

    summary = {
        "station_id": station_id,
        "filled_points": filled_count,
        "missing_points": int(len(missing_indices)),
        "status": "filled",
    }
    full_rows = []
    for idx, date_val in enumerate(pd.to_datetime(entry["dates"])):
        observed = entry["y_original_all"][idx]
        if pd.notna(observed):
            full_rows.append(
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
            item = fill_map.get(int(idx), {"final_streamflow": np.nan, "segment_length": 0, "fill_method": "UNFILLED", "quality_flag": "Q4"})
            full_rows.append(
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
    return pd.DataFrame(full_rows), summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build GSIM-PLUS product with single or hybrid imputation schemes.")
    parser.add_argument(
        "--scheme",
        choices=["maml_only", "maml_donor_trend_only", "maml_donor_trend_guarded", "donor_trend_only", "maml_calibrated_only", "hybrid_v1", "hybrid_v2", "all"],
        default="maml_only",
        help="Imputation scheme to run. 'all' generates all configured schemes.",
    )
    return parser.parse_args()


def scheme_output_dir(scheme_name):
    if scheme_name == "maml_only":
        return STEP8_DIR
    return STEP8_DIR / scheme_name


def production_method_label(scheme_rules):
    method_set = set(scheme_rules.values())
    if len(method_set) == 1:
        return next(iter(method_set))
    return "Hybrid(" + ", ".join(f"{flag}={scheme_rules[flag]}" for flag in ["Q1", "Q2", "Q3"]) + ")"


def run_scheme(scheme_name, scheme_rules, anchor_data, target_ids, similarity_df, models):
    output_dir = scheme_output_dir(scheme_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_low_flow = scheme_name == "maml_donor_trend_guarded"

    summaries = []
    gsim_fill_dir = output_dir / "GSIM_fill"
    gsim_fill_dir.mkdir(parents=True, exist_ok=True)

    full_series_path = output_dir / "gsim_plus_1995_2015_full_series.csv"
    infilled_points_path = output_dir / "gsim_plus_1995_2015_infilled_points.csv"

    if full_series_path.exists():
        full_series_path.unlink()
    if infilled_points_path.exists():
        infilled_points_path.unlink()

    full_written = False
    infilled_written = False
    total_filled_points = 0
    for i, station_id in enumerate(target_ids, start=1):
        if i % 500 == 0:
            print(f"  [{scheme_name}] Processed {i}/{len(target_ids)} target stations...")
        try:
            station_df, summary = fill_station(
                station_id,
                anchor_data,
                similarity_df,
                models,
                scheme_rules,
                guard_low_flow=guard_low_flow,
            )
            summaries.append(summary)
            if station_df is not None:
                station_path = gsim_fill_dir / f"{station_id}.csv"
                station_df.to_csv(station_path, index=False)

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
    summary_df.to_csv(output_dir / "gsim_plus_fill_summary.csv", index=False)

    product_summary = {
        "scheme_name": scheme_name,
        "scheme_rules": scheme_rules,
        "n_target_stations": int(len(target_ids)),
        "n_processed_stations": int(len(summary_df)),
        "n_filled_points": int(total_filled_points) if full_written else 0,
        "study_period": f"{STUDY_START_YEAR}-{STUDY_END_YEAR}",
        "production_method": production_method_label(scheme_rules) + (" + low_flow_guard" if guard_low_flow else ""),
        "station_split_dir": str(gsim_fill_dir),
    }

    with open(output_dir / "gsim_plus_product_summary.json", "w", encoding="utf-8") as f:
        json.dump(product_summary, f, indent=2)

    print(json.dumps(product_summary, indent=2))
    print(f"Saved to: {output_dir}")
    return product_summary


def main():
    args = parse_args()
    anchor_ids = pd.read_csv(STEP1_DIR / "anchor_stations.csv")["station_id"].tolist()
    target_ids = pd.read_csv(STEP1_DIR / "target_stations.csv")["station_id"].tolist()
    similarity_df = pd.read_csv(STEP3_DIR / "top_5_similar_stations.csv")

    print("Loading anchor data...")
    anchor_data = load_anchor_data(anchor_ids)
    print("Loading target task data...")
    target_base_data = {
        station_id: build_target_base_data(station_id)
        for station_id in target_ids
    }
    print("Training reusable models...")
    method_names = sorted(
        set(
            method
            for scheme_name, scheme_rules in (list(HYBRID_SCHEMES.items()) if args.scheme == "all" else [(args.scheme, HYBRID_SCHEMES[args.scheme])])
            for method in (
                list(scheme_rules.values()) + (["MAML"] if scheme_name == "maml_donor_trend_guarded" else [])
            )
        )
    )
    models = train_reusable_models(
        anchor_data,
        similarity_df,
        task_base_data=target_base_data,
        method_names=method_names,
    )

    if args.scheme == "all":
        selected_schemes = list(HYBRID_SCHEMES.items())
    else:
        selected_schemes = [(args.scheme, HYBRID_SCHEMES[args.scheme])]

    all_summaries = []
    for scheme_name, scheme_rules in selected_schemes:
        print(f"Running scheme: {scheme_name} -> {scheme_rules}")
        all_summaries.append(run_scheme(scheme_name, scheme_rules, anchor_data, target_ids, similarity_df, models))

    if len(all_summaries) > 1:
        with open(STEP8_DIR / "hybrid_scheme_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
