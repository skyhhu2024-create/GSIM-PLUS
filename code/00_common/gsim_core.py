import copy
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from gsim_plus_config import RANDOM_SEED


warnings.filterwarnings("ignore")


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(RANDOM_SEED)


def to_std(values, flow_mean, flow_std):
    values = np.asarray(values, dtype=float)
    if np.isfinite(flow_std) and flow_std > 0:
        return (values - flow_mean) / flow_std
    return values.copy()


def from_std(values, flow_mean, flow_std):
    values = np.asarray(values, dtype=float)
    if np.isfinite(flow_std) and flow_std > 0:
        return values * flow_std + flow_mean
    return values.copy()


def station_seed(station_id, scenario_code):
    return RANDOM_SEED + scenario_code * 1000 + sum(ord(ch) for ch in str(station_id))


def build_validation_entry(station_id, base_data, hide_indices):
    y_original = base_data["y_original"].copy()
    valid_mask = base_data["valid"].copy()
    train_mask = valid_mask.copy()
    train_mask[hide_indices] = False
    if train_mask.sum() < 12:
        return None

    train_values = y_original[train_mask]
    flow_mean = float(np.nanmean(train_values))
    flow_std = float(pd.Series(train_values).std())
    std_series_init = np.full(len(y_original), np.nan, dtype=float)
    std_series_init[train_mask] = to_std(y_original[train_mask], flow_mean, flow_std)

    return {
        "station_id": station_id,
        "dates": base_data["dates"],
        "year": base_data["year"],
        "month": base_data["month"],
        "month_sin": base_data["month_sin"],
        "month_cos": base_data["month_cos"],
        "y_original_all": y_original,
        "valid_mask": valid_mask,
        "train_mask": train_mask,
        "hide_indices": np.array(sorted(hide_indices), dtype=int),
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "std_series_init": std_series_init,
    }


def create_gap_rate_validation_set(target_base_data, gap_rate):
    validation_set = {}
    scenario_code = int(round(gap_rate * 100))
    for station_id, data in target_base_data.items():
        valid_indices = np.where(data["valid"])[0]
        if len(valid_indices) < 12:
            continue
        n_hide = max(1, int(len(valid_indices) * gap_rate))
        if len(valid_indices) - n_hide < 12:
            continue

        rng = np.random.default_rng(station_seed(station_id, scenario_code))
        hide_indices = np.sort(rng.choice(valid_indices, n_hide, replace=False))
        entry = build_validation_entry(station_id, data, hide_indices)
        if entry is not None:
            validation_set[station_id] = entry
    return validation_set


def build_feature_row(val_data, std_series, idx):
    lag1 = std_series[idx - 1] if idx > 0 and np.isfinite(std_series[idx - 1]) else 0.0
    return np.array([val_data["month_sin"][idx], val_data["month_cos"][idx], lag1], dtype=float)


def build_sequence_window(val_data, std_series, end_idx, seq_len):
    rows = []
    for idx in range(end_idx - seq_len, end_idx):
        rows.append(build_feature_row(val_data, std_series, idx))
    return np.array(rows, dtype=float)


def _build_target_tasks(similarity_df):
    target_tasks = defaultdict(list)
    for _, row in similarity_df.iterrows():
        target_tasks[row["target_station"]].append(
            {"anchor": row["anchor_station"], "similarity": float(row["similarity"])}
        )
    for station_id in target_tasks:
        target_tasks[station_id] = sorted(
            target_tasks[station_id],
            key=lambda item: item["similarity"],
            reverse=True,
        )
    return target_tasks


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return np.nan
    if np.all(weights <= 0):
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _weighted_std(values, weights, mean_value):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) <= 1:
        return 0.0
    if np.all(weights <= 0):
        return float(np.std(values))
    variance = np.average((values - mean_value) ** 2, weights=weights)
    return float(np.sqrt(max(variance, 0.0)))


def _month_climatology_std(val_data):
    climatology = {}
    train_indices = np.where(val_data["train_mask"])[0]
    for month in range(1, 13):
        month_idx = train_indices[val_data["month"][train_indices] == month]
        month_vals = val_data["std_series_init"][month_idx]
        month_vals = month_vals[np.isfinite(month_vals)]
        climatology[month] = float(month_vals.mean()) if len(month_vals) else 0.0
    return climatology


def prepare_task_entry(val_data, anchor_data, target_task_list):
    prepared = val_data.copy()
    month_climatology = _month_climatology_std(val_data)

    prior_std = np.zeros(len(val_data["year"]), dtype=float)
    donor_spread = np.zeros(len(val_data["year"]), dtype=float)
    donor_coverage = np.zeros(len(val_data["year"]), dtype=float)

    n_donors = max(len(target_task_list), 1)
    for idx in range(len(val_data["year"])):
        year = val_data["year"][idx]
        month = val_data["month"][idx]
        donor_values = []
        donor_weights = []
        for sim_info in target_task_list:
            aid = sim_info["anchor"]
            if aid not in anchor_data:
                continue
            anchor_info = anchor_data[aid]
            mask = (
                anchor_info["valid"]
                & (anchor_info["year"] == year)
                & (anchor_info["month"] == month)
            )
            if not mask.any():
                continue
            donor_orig = float(anchor_info["y_original"][mask][0])
            donor_target_std = float(to_std(donor_orig, val_data["flow_mean"], val_data["flow_std"]))
            if np.isfinite(donor_target_std):
                donor_values.append(donor_target_std)
                donor_weights.append(max(float(sim_info["similarity"]), 1e-6))

        if donor_values:
            prior = _weighted_mean(donor_values, donor_weights)
            spread = _weighted_std(donor_values, donor_weights, prior)
            coverage = len(donor_values) / n_donors
        else:
            prior = month_climatology.get(int(month), 0.0)
            spread = 0.0
            coverage = 0.0

        prior_std[idx] = prior
        donor_spread[idx] = spread
        donor_coverage[idx] = coverage

    prepared["prior_std"] = prior_std
    prepared["donor_spread"] = donor_spread
    prepared["donor_coverage"] = donor_coverage
    return prepared


def build_maml_feature_row(val_data, std_series, idx):
    lag1 = std_series[idx - 1] if idx > 0 and np.isfinite(std_series[idx - 1]) else 0.0
    return np.array(
        [
            val_data["month_sin"][idx],
            val_data["month_cos"][idx],
            lag1,
            val_data["prior_std"][idx],
            val_data["donor_spread"][idx],
            val_data["donor_coverage"][idx],
        ],
        dtype=float,
    )


def recursive_predict_with_sklearn(model, val_data):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    for idx in val_data["hide_indices"]:
        x_row = build_feature_row(val_data, std_series, idx).reshape(1, -1)
        pred_std = float(model.predict(x_row)[0])
        pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
        true_orig = val_data["y_original_all"][idx]
        if np.isfinite(true_orig) and np.isfinite(pred_orig):
            predictions[(val_data["station_id"], int(idx))] = {"true": float(true_orig), "pred": pred_orig}
            std_series[idx] = pred_std
    return predictions


def recursive_predict_with_maml(model, val_data, device):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    with torch.no_grad():
        for idx in val_data["hide_indices"]:
            x_row = build_feature_row(val_data, std_series, idx)
            pred_std = float(model(torch.FloatTensor(x_row).unsqueeze(0).to(device)).cpu().numpy()[0, 0])
            pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
            true_orig = val_data["y_original_all"][idx]
            if np.isfinite(true_orig) and np.isfinite(pred_orig):
                predictions[(val_data["station_id"], int(idx))] = {"true": float(true_orig), "pred": pred_orig}
                std_series[idx] = pred_std
    return predictions


def _identity_calibration():
    return {
        "slope": 1.0,
        "intercept": 0.0,
        "enabled": False,
        "n_points": 0,
    }


def apply_linear_calibration(pred_std, calibration):
    if not calibration or not calibration.get("enabled", False):
        return float(pred_std)
    return float(calibration["slope"] * pred_std + calibration["intercept"])


def fit_maml_station_calibration(
    model,
    val_data,
    device,
    min_points=12,
    slope_bounds=(0.8, 1.2),
    intercept_bounds=(-0.5, 0.5),
):
    train_indices = np.where(val_data["train_mask"])[0]
    if len(train_indices) < min_points:
        return _identity_calibration()

    pred_std, true_std = [], []
    std_series = val_data["std_series_init"]
    with torch.no_grad():
        for idx in train_indices:
            true_val = std_series[idx]
            if not np.isfinite(true_val):
                continue
            x_row = build_feature_row(val_data, std_series, idx)
            pred_val = float(model(torch.FloatTensor(x_row).unsqueeze(0).to(device)).cpu().numpy()[0, 0])
            if np.isfinite(pred_val):
                pred_std.append(pred_val)
                true_std.append(float(true_val))

    if len(pred_std) < min_points:
        return _identity_calibration()

    pred_std = np.asarray(pred_std, dtype=float)
    true_std = np.asarray(true_std, dtype=float)
    pred_mean = float(np.mean(pred_std))
    true_mean = float(np.mean(true_std))
    denom = float(np.sum((pred_std - pred_mean) ** 2))

    if denom <= 1e-8:
        raw_slope = 1.0
        raw_intercept = true_mean - pred_mean
    else:
        raw_slope = float(np.sum((pred_std - pred_mean) * (true_std - true_mean)) / denom)
        raw_intercept = float(true_mean - raw_slope * pred_mean)

    shrink = len(pred_std) / (len(pred_std) + 24.0)
    slope = 1.0 + shrink * (raw_slope - 1.0)
    intercept = shrink * raw_intercept
    slope = float(np.clip(slope, slope_bounds[0], slope_bounds[1]))
    intercept = float(np.clip(intercept, intercept_bounds[0], intercept_bounds[1]))

    base_rmse = float(np.sqrt(np.mean((true_std - pred_std) ** 2)))
    calibrated_pred = slope * pred_std + intercept
    cal_rmse = float(np.sqrt(np.mean((true_std - calibrated_pred) ** 2)))
    if cal_rmse >= base_rmse:
        return _identity_calibration()

    return {
        "slope": slope,
        "intercept": intercept,
        "enabled": True,
        "n_points": int(len(pred_std)),
        "base_rmse": base_rmse,
        "cal_rmse": cal_rmse,
    }


def recursive_predict_with_maml_calibrated(model, val_data, device, calibration):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    with torch.no_grad():
        for idx in val_data["hide_indices"]:
            x_row = build_feature_row(val_data, std_series, idx)
            pred_std = float(model(torch.FloatTensor(x_row).unsqueeze(0).to(device)).cpu().numpy()[0, 0])
            pred_std = apply_linear_calibration(pred_std, calibration)
            pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
            true_orig = val_data["y_original_all"][idx]
            if np.isfinite(true_orig) and np.isfinite(pred_orig):
                predictions[(val_data["station_id"], int(idx))] = {"true": float(true_orig), "pred": pred_orig}
                std_series[idx] = pred_std
    return predictions


def recursive_predict_with_lstm(model, val_data, device, seq_len):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    with torch.no_grad():
        for idx in val_data["hide_indices"]:
            if idx < seq_len:
                continue
            x_window = build_sequence_window(val_data, std_series, idx, seq_len)
            pred_std = float(model(torch.FloatTensor(x_window).unsqueeze(0).to(device)).cpu().numpy()[0, 0])
            pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
            true_orig = val_data["y_original_all"][idx]
            if np.isfinite(true_orig) and np.isfinite(pred_orig):
                predictions[(val_data["station_id"], int(idx))] = {"true": float(true_orig), "pred": pred_orig}
                std_series[idx] = pred_std
    return predictions


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"NSE": np.nan, "KGE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "n": 0}

    mean_obs = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_obs) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else 0
    beta = np.mean(y_pred) / mean_obs if mean_obs != 0 else 0
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return {
        "NSE": round(float(nse), 4),
        "KGE": round(float(kge), 4),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        "Bias": round(float((np.mean(y_pred) - mean_obs) / mean_obs) if mean_obs != 0 else np.nan, 4),
        "n": int(len(y_true)),
    }


def method_idw(anchor_data, validation_set, similarity_df):
    target_sim = _build_target_tasks(similarity_df)

    predictions = {}
    for target_id, val_data in validation_set.items():
        for idx in val_data["hide_indices"]:
            month = val_data["month"][idx]
            year = val_data["year"][idx]
            weights, values = [], []
            for sim_info in target_sim.get(target_id, []):
                aid = sim_info["anchor"]
                if aid not in anchor_data:
                    continue
                mask = (
                    anchor_data[aid]["valid"]
                    & (anchor_data[aid]["year"] == year)
                    & (anchor_data[aid]["month"] == month)
                )
                if mask.any():
                    val = anchor_data[aid]["y"][mask][0]
                    if np.isfinite(val):
                        dist = 1 - sim_info["similarity"]
                        weights.append(1 / (dist ** 2 + 0.001))
                        values.append(val)
            if weights:
                pred_std = float(np.average(values, weights=weights))
                pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
                true_orig = val_data["y_original_all"][idx]
                if np.isfinite(true_orig):
                    predictions[(target_id, int(idx))] = {"true": float(true_orig), "pred": pred_orig}
    return predictions


def method_baseline(anchor_data, validation_set, similarity_df):
    target_sim = _build_target_tasks(similarity_df)

    predictions = {}
    for target_id, val_data in validation_set.items():
        for idx in val_data["hide_indices"]:
            month = val_data["month"][idx]
            year = val_data["year"][idx]
            weighted_sum, weight_total = 0.0, 0.0
            for sim_info in target_sim.get(target_id, []):
                aid = sim_info["anchor"]
                if aid not in anchor_data:
                    continue
                mask = (
                    anchor_data[aid]["valid"]
                    & (anchor_data[aid]["year"] == year)
                    & (anchor_data[aid]["month"] == month)
                )
                if mask.any():
                    val = anchor_data[aid]["y"][mask][0]
                    if np.isfinite(val):
                        weighted_sum += sim_info["similarity"] * val
                        weight_total += sim_info["similarity"]
            if weight_total > 0:
                pred_std = weighted_sum / weight_total
                pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
                true_orig = val_data["y_original_all"][idx]
                if np.isfinite(true_orig):
                    predictions[(target_id, int(idx))] = {"true": float(true_orig), "pred": pred_orig}
    return predictions


def method_seasonal(anchor_data, validation_set, similarity_df):
    target_sim = _build_target_tasks(similarity_df)

    predictions = {}
    for target_id, val_data in validation_set.items():
        sim_list = target_sim.get(target_id, [])
        for idx in val_data["hide_indices"]:
            month = val_data["month"][idx]
            values = []
            for sim_info in sim_list[:5]:
                aid = sim_info["anchor"]
                if aid in anchor_data:
                    month_mask = anchor_data[aid]["month"] == month
                    if month_mask.any():
                        month_vals = anchor_data[aid]["y"][month_mask]
                        values.extend(month_vals[np.isfinite(month_vals)])
            if values:
                pred_std = float(np.mean(values))
                pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
                true_orig = val_data["y_original_all"][idx]
                if np.isfinite(true_orig):
                    predictions[(target_id, int(idx))] = {"true": float(true_orig), "pred": pred_orig}
    return predictions


def _previous_year_month(year, month):
    if int(month) == 1:
        return int(year) - 1, 12
    return int(year), int(month) - 1


def _donor_std_at(anchor_info, year, month, target_flow_mean, target_flow_std):
    mask = (
        anchor_info["valid"]
        & (anchor_info["year"] == year)
        & (anchor_info["month"] == month)
    )
    if not mask.any():
        return np.nan
    donor_orig = float(anchor_info["y_original"][mask][0])
    donor_std = float(to_std(donor_orig, target_flow_mean, target_flow_std))
    return donor_std if np.isfinite(donor_std) else np.nan


def build_dtrr_feature_row(anchor_data, target_task_list, val_data, std_series, idx, top_k=3):
    lag1 = std_series[idx - 1] if idx > 0 and np.isfinite(std_series[idx - 1]) else 0.0
    year = int(val_data["year"][idx])
    month = int(val_data["month"][idx])
    prev_year, prev_month = _previous_year_month(year, month)

    donor_current_vals = []
    donor_delta_vals = []
    donor_weights = []
    donor_features = []

    for sim_info in target_task_list[:top_k]:
        aid = sim_info["anchor"]
        if aid not in anchor_data:
            donor_features.extend([0.0, 0.0])
            continue

        anchor_info = anchor_data[aid]
        cur_std = _donor_std_at(anchor_info, year, month, val_data["flow_mean"], val_data["flow_std"])
        prev_std = _donor_std_at(anchor_info, prev_year, prev_month, val_data["flow_mean"], val_data["flow_std"])

        cur_feat = float(cur_std) if np.isfinite(cur_std) else 0.0
        delta_feat = float(cur_std - prev_std) if np.isfinite(cur_std) and np.isfinite(prev_std) else 0.0
        donor_features.extend([cur_feat, delta_feat])

        if np.isfinite(cur_std):
            donor_current_vals.append(float(cur_std))
            donor_delta_vals.append(delta_feat)
            donor_weights.append(max(float(sim_info["similarity"]), 1e-6))

    weighted_current = _weighted_mean(donor_current_vals, donor_weights) if donor_current_vals else 0.0
    weighted_delta = _weighted_mean(donor_delta_vals, donor_weights) if donor_delta_vals else 0.0
    donor_coverage = len(donor_current_vals) / max(min(len(target_task_list), top_k), 1)

    feature_row = np.array(
        [
            val_data["month_sin"][idx],
            val_data["month_cos"][idx],
            lag1,
            *donor_features,
            weighted_current,
            weighted_delta,
            donor_coverage,
        ],
        dtype=float,
    )
    return feature_row


def fallback_dtrr_std(anchor_data, target_task_list, val_data, std_series, idx):
    feature_row = build_dtrr_feature_row(anchor_data, target_task_list, val_data, std_series, idx)
    donor_block = feature_row[3:-3]
    donor_currents = donor_block[0::2]
    weighted_current = float(feature_row[-3])
    lag1 = float(feature_row[2])
    if np.any(np.abs(donor_currents) > 0):
        return weighted_current
    return lag1


def fit_dtrr_model(anchor_data, target_task_list, val_data, top_k=3, min_points=12, alpha=2.0):
    train_indices = np.where(val_data["train_mask"])[0]
    std_series = val_data["std_series_init"]
    X_train, y_train = [], []

    for idx in train_indices:
        y_std = std_series[idx]
        if not np.isfinite(y_std):
            continue
        row = build_dtrr_feature_row(anchor_data, target_task_list, val_data, std_series, idx, top_k=top_k)
        if row[-1] <= 0 and not np.isfinite(row[2]):
            continue
        X_train.append(row)
        y_train.append(float(y_std))

    if len(X_train) < min_points:
        return None

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    y_q01 = float(np.quantile(y_train, 0.01))
    y_q99 = float(np.quantile(y_train, 0.99))
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    model._gsim_top_k = int(top_k)
    model._gsim_y_low = min(y_q01, y_min)
    model._gsim_y_high = max(y_q99, y_max)
    model._gsim_y_span = max(model._gsim_y_high - model._gsim_y_low, 1.0)
    return model


def predict_dtrr_std_raw(model, anchor_data, target_task_list, val_data, std_series, idx, top_k=None):
    if top_k is None:
        top_k = int(getattr(model, "_gsim_top_k", 3)) if model is not None else 3
    feature_row = build_dtrr_feature_row(anchor_data, target_task_list, val_data, std_series, idx, top_k=top_k)
    if model is None:
        return fallback_dtrr_std(anchor_data, target_task_list, val_data, std_series, idx)
    return float(model.predict(feature_row.reshape(1, -1))[0])


def recursive_predict_with_dtrr(model, anchor_data, target_task_list, val_data, top_k=3):
    predictions = {}
    std_series = val_data["std_series_init"].copy()
    for idx in val_data["hide_indices"]:
        idx = int(idx)
        pred_std = predict_dtrr_std_raw(
            model,
            anchor_data,
            target_task_list,
            val_data,
            std_series,
            idx,
            top_k=top_k,
        )
        pred_orig = float(from_std(pred_std, val_data["flow_mean"], val_data["flow_std"]))
        true_orig = val_data["y_original_all"][idx]
        if np.isfinite(true_orig) and np.isfinite(pred_orig):
            predictions[(val_data["station_id"], int(idx))] = {"true": float(true_orig), "pred": pred_orig}
            std_series[idx] = pred_std
    return predictions


def method_dtrr(anchor_data, validation_set, similarity_df):
    target_tasks = _build_target_tasks(similarity_df)
    predictions = {}
    for target_id, val_data in validation_set.items():
        task_list = target_tasks.get(target_id, [])
        if not task_list:
            continue
        model = fit_dtrr_model(anchor_data, task_list, val_data)
        predictions.update(recursive_predict_with_dtrr(model, anchor_data, task_list, val_data))
    return predictions


def stabilize_dtrr_std(raw_pred, feature_row, model=None, recursive_depth=0):
    pred_std = float(raw_pred)
    if not np.isfinite(pred_std):
        return np.nan

    if model is not None:
        y_low = getattr(model, "_gsim_y_low", None)
        y_high = getattr(model, "_gsim_y_high", None)
        y_span = float(getattr(model, "_gsim_y_span", 1.0))
        if y_low is not None and y_high is not None:
            train_margin = max(1.0, 0.15 * y_span)
            pred_std = float(np.clip(pred_std, y_low - train_margin, y_high + train_margin))

    lag1 = float(feature_row[2])
    donor_block = np.asarray(feature_row[3:-3], dtype=float)
    donor_currents = donor_block[0::2]
    donor_currents = donor_currents[np.isfinite(donor_currents)]
    weighted_current = float(feature_row[-3])
    weighted_delta = float(feature_row[-2])
    donor_coverage = float(feature_row[-1])

    if donor_coverage > 0 and donor_currents.size > 0:
        donor_margin = max(1.5, 1.5 * float(np.std(donor_currents)), abs(weighted_delta) + 1.0)
        donor_low = float(np.min(donor_currents) - donor_margin)
        donor_high = float(np.max(donor_currents) + donor_margin)
        pred_std = float(np.clip(pred_std, donor_low, donor_high))

        if recursive_depth > 0 and np.isfinite(weighted_current):
            donor_blend = min(0.2 + 0.08 * recursive_depth, 0.6)
            pred_std = float((1.0 - donor_blend) * pred_std + donor_blend * weighted_current)

    if np.isfinite(lag1):
        jump_cap = max(2.0, 2.5 * abs(weighted_delta) + 1.0)
        if recursive_depth > 0:
            jump_cap *= 0.75
        pred_std = float(np.clip(pred_std, lag1 - jump_cap, lag1 + jump_cap))

    return pred_std


def predict_dtrr_std(model, anchor_data, target_task_list, val_data, std_series, idx, top_k=None, recursive_depth=0):
    if top_k is None:
        top_k = int(getattr(model, "_gsim_top_k", 3)) if model is not None else 3
    feature_row = build_dtrr_feature_row(anchor_data, target_task_list, val_data, std_series, idx, top_k=top_k)
    if model is None:
        raw_pred = fallback_dtrr_std(anchor_data, target_task_list, val_data, std_series, idx)
    else:
        raw_pred = float(model.predict(feature_row.reshape(1, -1))[0])
    return stabilize_dtrr_std(raw_pred, feature_row, model=model, recursive_depth=recursive_depth)


def train_ml_model(anchor_data, model_type):
    print(f"Training {model_type} model on anchor data...")
    X_train, y_train = [], []
    for data in anchor_data.values():
        valid_mask = data["valid"]
        if valid_mask.sum() > 12:
            X_train.extend(data["X"][valid_mask])
            y_train.extend(data["y"][valid_mask])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    print(f"  {model_type} training complete: {len(X_train)} samples")
    return model


def method_ml(anchor_data, validation_set, model_type, trained_model=None):
    model = train_ml_model(anchor_data, model_type) if trained_model is None else trained_model
    predictions = {}
    for val_data in validation_set.values():
        predictions.update(recursive_predict_with_sklearn(model, val_data))
    return predictions


class MAMLModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def _extract_base_from_validation(validation_set):
    task_base = {}
    for station_id, val_data in validation_set.items():
        task_base[station_id] = {
            "dates": val_data["dates"],
            "year": val_data["year"],
            "month": val_data["month"],
            "month_sin": val_data["month_sin"],
            "month_cos": val_data["month_cos"],
            "y_original": val_data["y_original_all"],
            "valid": val_data["valid_mask"],
        }
    return task_base


def _sample_meta_hide_indices(base_data, rng):
    valid_mask = base_data["valid"]
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 24:
        return None

    scenario = rng.choice(["random30", "block3", "block6", "block12"], p=[0.35, 0.25, 0.20, 0.20])
    if scenario == "random30":
        n_hide = max(1, int(len(valid_indices) * 0.30))
        if len(valid_indices) - n_hide < 12:
            return None
        return np.sort(rng.choice(valid_indices, n_hide, replace=False))

    gap_length = int(scenario.replace("block", ""))
    starts = []
    for start_idx in range(len(valid_mask) - gap_length + 1):
        if valid_mask[start_idx : start_idx + gap_length].all():
            starts.append(start_idx)
    if not starts:
        return None
    start_idx = int(rng.choice(starts))
    return np.arange(start_idx, start_idx + gap_length, dtype=int)


def build_anchor_support_tensors(anchor_data, target_task_list, device, min_points=6):
    support_X_list, support_y_list = [], []
    for sim_info in target_task_list:
        aid = sim_info["anchor"]
        if aid not in anchor_data:
            continue
        anchor_info = anchor_data[aid]
        valid_mask = anchor_info["valid"]
        if valid_mask.sum() <= min_points:
            continue
        support_X_list.append(torch.FloatTensor(anchor_info["X"][valid_mask]).to(device))
        support_y_list.append(torch.FloatTensor(anchor_info["y"][valid_mask]).reshape(-1, 1).to(device))

    if not support_X_list:
        return None
    return torch.cat(support_X_list, dim=0), torch.cat(support_y_list, dim=0)


def build_self_support_tensors(val_data, device, min_points=6):
    train_indices = np.where(val_data["train_mask"])[0]
    if len(train_indices) <= min_points:
        return None

    std_series = val_data["std_series_init"]
    support_X, support_y = [], []
    for idx in train_indices:
        y_std = std_series[idx]
        if not np.isfinite(y_std):
            continue
        support_X.append(build_feature_row(val_data, std_series, idx))
        support_y.append(y_std)

    if len(support_X) <= min_points:
        return None

    support_X = torch.FloatTensor(np.array(support_X, dtype=float)).to(device)
    support_y = torch.FloatTensor(np.array(support_y, dtype=float)).reshape(-1, 1).to(device)
    return support_X, support_y


def build_query_tensors(val_data, device):
    std_true = to_std(val_data["y_original_all"], val_data["flow_mean"], val_data["flow_std"])
    query_X, query_y = [], []
    std_series = val_data["std_series_init"].copy()
    for idx in val_data["hide_indices"]:
        true_std = std_true[idx]
        if not np.isfinite(true_std):
            continue
        query_X.append(build_feature_row(val_data, std_series, idx))
        query_y.append(true_std)
    if not query_X:
        return None
    query_X = torch.FloatTensor(np.array(query_X, dtype=float)).to(device)
    query_y = torch.FloatTensor(np.array(query_y, dtype=float)).reshape(-1, 1).to(device)
    return query_X, query_y


def _weighted_mse(pred, target, weights=None):
    if weights is None:
        return nn.MSELoss()(pred, target)
    weight_sum = torch.clamp(weights.sum(), min=1e-6)
    return torch.sum(weights * (pred - target) ** 2) / weight_sum


def _split_support_tuple(support):
    if support is None:
        return None, None, None
    if len(support) == 2:
        return support[0], support[1], None
    return support


def adapt_maml_model(base_model, anchor_support, self_support, inner_lr, inner_steps):
    support_batches = []
    if anchor_support is not None:
        support_batches.append(anchor_support)
    if self_support is not None:
        support_batches.append(self_support)
    if not support_batches:
        return None

    adapted_model = copy.deepcopy(base_model)
    inner_opt = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    adapted_model.train()

    if len(support_batches) == 1:
        step_plan = [inner_steps]
    else:
        total_steps = max(inner_steps, 2)
        anchor_steps = max(1, total_steps // 2)
        self_steps = max(1, total_steps - anchor_steps)
        step_plan = [anchor_steps, self_steps]

    for support, n_steps in zip(support_batches, step_plan):
        support_X, support_y, support_w = _split_support_tuple(support)
        for _ in range(n_steps):
            inner_opt.zero_grad()
            loss = _weighted_mse(adapted_model(support_X), support_y, support_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            inner_opt.step()
    adapted_model.eval()
    return adapted_model


def train_maml_model(anchor_data, similarity_df, task_base_data, config):
    print("Training MAML meta-model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAMLModel(
        input_dim=config.get("input_dim", 3),
        hidden_dim=config["hidden_dim"],
    ).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=config["meta_lr"])

    tasks = []
    for _, row in similarity_df.iterrows():
        anchor_id = row["anchor_station"]
        if anchor_id in anchor_data:
            tasks.append({"anchor": anchor_id, "similarity": row["similarity"]})
    if not tasks:
        raise ValueError("No valid anchor-side tasks available for MAML training.")

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        batch_size = min(config["meta_batch_size"], len(tasks))
        batch_tasks = np.random.choice(len(tasks), batch_size, replace=False)

        for task_idx in batch_tasks:
            task = tasks[task_idx]
            anchor_info = anchor_data[task["anchor"]]
            valid_mask = anchor_info["valid"]
            if valid_mask.sum() < 12:
                continue

            X = torch.FloatTensor(anchor_info["X"][valid_mask]).to(device)
            y = torch.FloatTensor(anchor_info["y"][valid_mask]).reshape(-1, 1).to(device)
            n_support = len(X) // 2
            support_x, support_y = X[:n_support], y[:n_support]
            query_x, query_y = X[n_support:], y[n_support:]

            adapted_model = copy.deepcopy(model)
            inner_opt = optim.SGD(adapted_model.parameters(), lr=config["inner_lr"])
            for _ in range(config["inner_steps"]):
                inner_opt.zero_grad()
                loss = nn.MSELoss()(adapted_model(support_x), support_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
                inner_opt.step()

            task_loss = nn.MSELoss()(adapted_model(query_x), query_y)
            epoch_loss += task_loss

        if epoch_loss > 0:
            meta_optimizer.zero_grad()
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            meta_optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  MAML epoch {epoch + 1}/{config['epochs']}")

    print("  MAML training complete")
    return model


def method_maml(anchor_data, validation_set, similarity_df, trained_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml_config = {
        "meta_lr": 0.001,
        "inner_lr": 0.05,
        "inner_steps": 10,
        "meta_batch_size": 8,
        "epochs": 60,
        "hidden_dim": 64,
        "input_dim": 3,
    }

    model = train_maml_model(anchor_data, similarity_df, None, maml_config) if trained_model is None else trained_model
    model.eval()

    target_tasks = _build_target_tasks(similarity_df)
    predictions = {}
    for target_id, val_data in validation_set.items():
        if target_id not in target_tasks:
            continue
        anchor_support = build_anchor_support_tensors(anchor_data, target_tasks[target_id], device)
        self_support = build_self_support_tensors(val_data, device)
        adapted_model = adapt_maml_model(
            model,
            anchor_support,
            self_support,
            inner_lr=maml_config["inner_lr"],
            inner_steps=maml_config["inner_steps"],
        )
        if adapted_model is None:
            continue
        predictions.update(recursive_predict_with_maml(adapted_model, val_data, device))
    return predictions


def method_maml_calibrated(anchor_data, validation_set, similarity_df, trained_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml_config = {
        "meta_lr": 0.001,
        "inner_lr": 0.05,
        "inner_steps": 10,
        "meta_batch_size": 8,
        "epochs": 60,
        "hidden_dim": 64,
        "input_dim": 3,
    }

    model = train_maml_model(anchor_data, similarity_df, None, maml_config) if trained_model is None else trained_model
    model.eval()

    target_tasks = _build_target_tasks(similarity_df)
    predictions = {}
    for target_id, val_data in validation_set.items():
        if target_id not in target_tasks:
            continue
        anchor_support = build_anchor_support_tensors(anchor_data, target_tasks[target_id], device)
        self_support = build_self_support_tensors(val_data, device)
        adapted_model = adapt_maml_model(
            model,
            anchor_support,
            self_support,
            inner_lr=maml_config["inner_lr"],
            inner_steps=maml_config["inner_steps"],
        )
        if adapted_model is None:
            continue
        calibration = fit_maml_station_calibration(adapted_model, val_data, device)
        predictions.update(recursive_predict_with_maml_calibrated(adapted_model, val_data, device, calibration))
    return predictions


def train_lstm_model(anchor_data, seq_len=6):
    print("Training LSTM model on anchor data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq, y_seq = [], []

    for data in anchor_data.values():
        valid_mask = data["valid"]
        if valid_mask.sum() > seq_len + 5:
            X = data["X"][valid_mask]
            y = data["y"][valid_mask]
            for i in range(len(X) - seq_len):
                X_seq.append(X[i : i + seq_len])
                y_seq.append(y[i + seq_len])

    if len(X_seq) < 100:
        print("  LSTM skipped: not enough sequences")
        return None

    X_tensor = torch.FloatTensor(np.array(X_seq)).to(device)
    y_tensor = torch.FloatTensor(np.array(y_seq)).reshape(-1, 1).to(device)
    model = SimpleLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(30):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"  LSTM training complete: {len(X_seq)} sequences")
    return model


def method_lstm(anchor_data, validation_set, trained_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 6
    model = train_lstm_model(anchor_data, seq_len=seq_len) if trained_model is None else trained_model
    if model is None:
        return {}

    predictions = {}
    for val_data in validation_set.values():
        predictions.update(recursive_predict_with_lstm(model, val_data, device, seq_len))
    return predictions


def evaluate_on_common_points(all_predictions, method_names):
    common_keys = None
    for pred_dict in all_predictions:
        keys = set(pred_dict.keys())
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()
    if not common_keys:
        return {}, set()

    results = {}
    for name, pred_dict in zip(method_names, all_predictions):
        y_true = [pred_dict[k]["true"] for k in common_keys if k in pred_dict]
        y_pred = [pred_dict[k]["pred"] for k in common_keys if k in pred_dict]
        results[name] = calculate_metrics(y_true, y_pred)
    return results, common_keys


def build_prediction_dataframe(pred_dict, validation_set, common_keys):
    rows = []
    for target_id, idx in sorted(common_keys):
        pred = pred_dict.get((target_id, idx))
        if pred is None:
            continue
        val_data = validation_set[target_id]
        rows.append(
            {
                "target_station": target_id,
                "date": pd.to_datetime(val_data["dates"][idx]).strftime("%Y-%m-%d"),
                "year": int(val_data["year"][idx]),
                "month": int(val_data["month"][idx]),
                "true": float(pred["true"]),
                "pred": float(pred["pred"]),
            }
        )
    return pd.DataFrame(rows)
