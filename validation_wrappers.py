import hashlib
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import gsim_core as core

from gsim_plus_config import MODEL_CACHE_DIR, RANDOM_SEED
from gsim_plus_utils import read_station_study_period, resolve_station_file


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

DEFAULT_METHOD_ORDER = [
    "IDW",
    "SeasonalMean",
    "RandomForest",
    "Linear",
    "KNN",
    "MAML",
    "DTRR",
    "LSTM",
]
EXPERIMENTAL_METHODS = [
    "MAML_Calibrated",
    "Baseline_Ours",
]
METHOD_ORDER = DEFAULT_METHOD_ORDER + EXPERIMENTAL_METHODS

FOCUS_METHODS = ["MAML", "DTRR", "RandomForest", "IDW"]
TRAINED_MODEL_METHODS = {
    "RandomForest": "rf",
    "Linear": "linear",
    "KNN": "knn",
    "MAML": "maml",
    "MAML_Calibrated": "maml",
    "LSTM": "lstm",
}
MAML_CONFIG = {
    "meta_lr": 0.001,
    "inner_lr": 0.05,
    "inner_steps": 10,
    "meta_batch_size": 8,
    "epochs": 60,
    "hidden_dim": 64,
    "input_dim": 3,
}
LSTM_CONFIG = {
    "seq_len": 6,
    "input_dim": 3,
    "hidden_dim": 32,
    "output_dim": 1,
}


METHOD_ALIASES = {
    "MAML_Ours": "MAML",
}


def load_anchor_data(station_ids):
    data = {}
    for station_id in station_ids:
        try:
            df = read_station_study_period(station_id)
            if len(df) == 0 or "MEAN" not in df.columns:
                continue
            df["streamflow"] = pd.to_numeric(df["MEAN"], errors="coerce")
            flow_mean = df["streamflow"].mean()
            flow_std = df["streamflow"].std()
            df["streamflow_std"] = core.to_std(df["streamflow"].values, flow_mean, flow_std)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["flow_lag1"] = pd.Series(df["streamflow_std"]).shift(1)
            data[station_id] = {
                "dates": df["date"].values,
                "year": df["year"].values,
                "month": df["month"].values,
                "month_sin": df["month_sin"].values,
                "month_cos": df["month_cos"].values,
                "X": df[["month_sin", "month_cos", "flow_lag1"]].fillna(0).values,
                "y": df["streamflow_std"].values,
                "y_original": df["streamflow"].values,
                "valid": df["streamflow"].notna().values,
                "flow_mean": flow_mean,
                "flow_std": flow_std,
            }
        except Exception:
            continue
    return data


def load_target_base_data(station_ids):
    data = {}
    for station_id in station_ids:
        try:
            df = read_station_study_period(station_id)
            if len(df) == 0 or "MEAN" not in df.columns:
                continue
            df["streamflow"] = pd.to_numeric(df["MEAN"], errors="coerce")
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            data[station_id] = {
                "dates": df["date"].values,
                "year": df["year"].values,
                "month": df["month"].values,
                "month_sin": df["month_sin"].values,
                "month_cos": df["month_cos"].values,
                "y_original": df["streamflow"].values,
                "valid": df["streamflow"].notna().values,
            }
        except Exception:
            continue
    return data


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _similarity_signature(similarity_df):
    cols = [str(c) for c in similarity_df.columns]
    digest = hashlib.sha256()
    digest.update("|".join(cols).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(similarity_df, index=True).values.tobytes())
    return digest.hexdigest()


def _anchor_signature(anchor_data):
    digest = hashlib.sha256()
    for station_id in sorted(anchor_data):
        path = resolve_station_file(station_id)
        stat = path.stat()
        info = anchor_data[station_id]
        digest.update(
            f"{station_id}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}|{len(info['X'])}|{int(np.sum(info['valid']))}".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _cache_key(anchor_data, similarity_df):
    digest = hashlib.sha256()
    digest.update(_anchor_signature(anchor_data).encode("utf-8"))
    digest.update(_similarity_signature(similarity_df).encode("utf-8"))
    digest.update(json.dumps(MAML_CONFIG, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(LSTM_CONFIG, sort_keys=True).encode("utf-8"))
    digest.update(str(RANDOM_SEED).encode("utf-8"))
    return digest.hexdigest()[:16]


def _cache_paths(cache_dir):
    return {
        "manifest": cache_dir / "manifest.json",
        "linear": cache_dir / "linear.pkl",
        "rf": cache_dir / "rf.pkl",
        "knn": cache_dir / "knn.pkl",
        "maml": cache_dir / "maml.pt",
        "lstm": cache_dir / "lstm.pt",
        "lstm_none": cache_dir / "lstm_none.json",
    }


def _has_complete_cache(paths, required_model_types):
    if not paths["manifest"].exists():
        return False
    for model_type in required_model_types:
        if model_type == "lstm":
            if not (paths["lstm"].exists() or paths["lstm_none"].exists()):
                return False
            continue
        if not paths[model_type].exists():
            return False
    return True


def _save_manifest(paths, cache_key, anchor_data, similarity_df):
    manifest = {
        "cache_key": cache_key,
        "random_seed": RANDOM_SEED,
        "n_anchor_stations": len(anchor_data),
        "n_similarity_rows": int(len(similarity_df)),
        "maml_config": MAML_CONFIG,
        "lstm_config": LSTM_CONFIG,
    }
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _save_models(cache_dir, cache_key, anchor_data, similarity_df, models):
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir)

    if "linear" in models:
        with open(paths["linear"], "wb") as f:
            pickle.dump(models["linear"], f)
    if "rf" in models:
        with open(paths["rf"], "wb") as f:
            pickle.dump(models["rf"], f)
    if "knn" in models:
        with open(paths["knn"], "wb") as f:
            pickle.dump(models["knn"], f)

    if "maml" in models:
        torch.save(
            {
                "state_dict": models["maml"].state_dict(),
                "config": MAML_CONFIG,
            },
            paths["maml"],
        )

    if "lstm" in models:
        if models["lstm"] is None:
            with open(paths["lstm_none"], "w", encoding="utf-8") as f:
                json.dump({"model": None}, f)
        else:
            torch.save(
                {
                    "state_dict": models["lstm"].state_dict(),
                    "config": LSTM_CONFIG,
                },
                paths["lstm"],
            )
            if paths["lstm_none"].exists():
                paths["lstm_none"].unlink()

    _save_manifest(paths, cache_key, anchor_data, similarity_df)


def _load_models(cache_dir, required_model_types):
    device = _device()
    paths = _cache_paths(cache_dir)
    models = {}

    if "linear" in required_model_types:
        with open(paths["linear"], "rb") as f:
            models["linear"] = pickle.load(f)
    if "rf" in required_model_types:
        with open(paths["rf"], "rb") as f:
            models["rf"] = pickle.load(f)
    if "knn" in required_model_types:
        with open(paths["knn"], "rb") as f:
            models["knn"] = pickle.load(f)

    if "maml" in required_model_types:
        maml_payload = torch.load(paths["maml"], map_location=device)
        maml = core.MAMLModel(
            input_dim=maml_payload["config"].get("input_dim", 3),
            hidden_dim=maml_payload["config"]["hidden_dim"],
        ).to(device)
        maml.load_state_dict(maml_payload["state_dict"])
        maml.eval()
        models["maml"] = maml

    if "lstm" in required_model_types:
        if paths["lstm_none"].exists():
            models["lstm"] = None
        else:
            lstm_payload = torch.load(paths["lstm"], map_location=device)
            lstm = core.SimpleLSTM(
                input_dim=lstm_payload["config"]["input_dim"],
                hidden_dim=lstm_payload["config"]["hidden_dim"],
                output_dim=lstm_payload["config"]["output_dim"],
            ).to(device)
            lstm.load_state_dict(lstm_payload["state_dict"])
            lstm.eval()
            models["lstm"] = lstm

    return models


def _task_signature(task_base_data):
    digest = hashlib.sha256()
    for station_id in sorted(task_base_data):
        info = task_base_data[station_id]
        digest.update(
            f"{station_id}|{len(info['year'])}|{int(np.sum(info['valid']))}".encode("utf-8")
        )
    return digest.hexdigest()


def parse_method_selection(raw_methods):
    if not raw_methods:
        return DEFAULT_METHOD_ORDER.copy()
    if isinstance(raw_methods, str):
        requested = [item.strip() for item in raw_methods.split(",") if item.strip()]
    else:
        requested = list(raw_methods)
    requested = [METHOD_ALIASES.get(name, name) for name in requested]
    invalid = [name for name in requested if name not in METHOD_ORDER]
    if invalid:
        raise ValueError(f"Unsupported methods requested: {invalid}")
    return [name for name in METHOD_ORDER if name in requested]


def required_model_types_for_methods(method_names):
    return sorted({TRAINED_MODEL_METHODS[name] for name in method_names if name in TRAINED_MODEL_METHODS})


def train_reusable_models(
    anchor_data,
    similarity_df,
    task_base_data=None,
    force_retrain=False,
    method_names=None,
):
    if task_base_data is None:
        task_base_data = {}
    selected_methods = parse_method_selection(method_names)
    required_model_types = required_model_types_for_methods(selected_methods)
    cache_key = _cache_key(anchor_data, similarity_df)
    if task_base_data:
        cache_key = hashlib.sha256(
            f"{cache_key}|{_task_signature(task_base_data)}".encode("utf-8")
        ).hexdigest()[:16]
    if required_model_types:
        cache_key = hashlib.sha256(
            f"{cache_key}|{','.join(required_model_types)}".encode("utf-8")
        ).hexdigest()[:16]
    cache_dir = MODEL_CACHE_DIR / f"models_{cache_key}"
    paths = _cache_paths(cache_dir)

    if not required_model_types:
        return {}

    if not force_retrain and _has_complete_cache(paths, required_model_types):
        print(f"Loading reusable models from cache: {cache_dir}")
        return _load_models(cache_dir, required_model_types)

    if force_retrain:
        print("Force retrain enabled; ignoring model cache.")
    else:
        print(f"Model cache miss; training reusable models: {cache_dir}")

    models = {}
    if "linear" in required_model_types:
        models["linear"] = core.train_ml_model(anchor_data, "linear")
    if "rf" in required_model_types:
        models["rf"] = core.train_ml_model(anchor_data, "rf")
    if "knn" in required_model_types:
        models["knn"] = core.train_ml_model(anchor_data, "knn")
    if "maml" in required_model_types:
        models["maml"] = core.train_maml_model(anchor_data, similarity_df, task_base_data, MAML_CONFIG)
    if "lstm" in required_model_types:
        models["lstm"] = core.train_lstm_model(anchor_data, seq_len=LSTM_CONFIG["seq_len"])
    _save_models(cache_dir, cache_key, anchor_data, similarity_df, models)
    return models


def run_all_methods(anchor_data, validation_set, similarity_df, models, method_names=None):
    selected_methods = parse_method_selection(method_names)
    predictions = []
    for method_name in selected_methods:
        if method_name == "IDW":
            predictions.append(core.method_idw(anchor_data, validation_set, similarity_df))
        elif method_name == "Baseline_Ours":
            predictions.append(core.method_baseline(anchor_data, validation_set, similarity_df))
        elif method_name == "SeasonalMean":
            predictions.append(core.method_seasonal(anchor_data, validation_set, similarity_df))
        elif method_name == "RandomForest":
            predictions.append(core.method_ml(anchor_data, validation_set, "rf", trained_model=models["rf"]))
        elif method_name == "Linear":
            predictions.append(core.method_ml(anchor_data, validation_set, "linear", trained_model=models["linear"]))
        elif method_name == "KNN":
            predictions.append(core.method_ml(anchor_data, validation_set, "knn", trained_model=models["knn"]))
        elif method_name == "MAML":
            predictions.append(core.method_maml(anchor_data, validation_set, similarity_df, trained_model=models["maml"]))
        elif method_name == "DTRR":
            predictions.append(core.method_dtrr(anchor_data, validation_set, similarity_df))
        elif method_name == "MAML_Calibrated":
            predictions.append(core.method_maml_calibrated(anchor_data, validation_set, similarity_df, trained_model=models["maml"]))
        elif method_name == "LSTM":
            predictions.append(core.method_lstm(anchor_data, validation_set, trained_model=models["lstm"]))
    results, common_keys = core.evaluate_on_common_points(predictions, selected_methods)
    return predictions, results, common_keys, selected_methods


def save_method_outputs(output_dir, scenario_name, validation_set, all_predictions, results, common_keys, method_names):
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for method, pred_dict in zip(method_names, all_predictions):
        df = core.build_prediction_dataframe(pred_dict, validation_set, common_keys)
        df.to_csv(scenario_dir / f"{method}_predictions.csv", index=False)
        metric = results[method]
        comparison_rows.append(
            {
                "Method": method,
                "NSE": metric["NSE"],
                "KGE": metric["KGE"],
                "RMSE": metric["RMSE"],
                "MAE": metric["MAE"],
                "Bias": metric["Bias"],
                "n": metric["n"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("NSE", ascending=False)
    comparison_df.to_csv(scenario_dir / "unified_comparison_all_methods.csv", index=False)
    return comparison_df


def plot_summary_heatmap(summary_df, scenario_col, output_file, method_names=None):
    row_order = parse_method_selection(method_names) if method_names else DEFAULT_METHOD_ORDER
    pivot = summary_df.pivot(index="method", columns=scenario_col, values="NSE").reindex(row_order)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_file)
    plt.close(fig)


def plot_scatter_panel(scenario_outputs, scenario_names, output_file):
    available = []
    for method in FOCUS_METHODS:
        if all(method in scenario_outputs[scenario]["predictions"] for scenario in scenario_names):
            available.append(method)
    if not available:
        sample = scenario_outputs[scenario_names[0]]["predictions"]
        available = list(sample.keys())
    fig, axes = plt.subplots(len(available), len(scenario_names), figsize=(10.5, 10), squeeze=False)
    colors = {
        "SeasonalMean": "#8fb339",
        "MAML": "#c8553d",
        "DTRR": "#d1495b",
        "MAML_Calibrated": "#b56576",
        "RandomForest": "#264653",
        "IDW": "#457b9d",
    }
    for r, method in enumerate(available):
        for c, scenario in enumerate(scenario_names):
            output = scenario_outputs[scenario]
            pred_dict = output["predictions"][method]
            keys = sorted(output["common_keys"])
            y_true = np.array([pred_dict[k]["true"] for k in keys], dtype=float)
            y_pred = np.array([pred_dict[k]["pred"] for k in keys], dtype=float)
            ax = axes[r, c]
            ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=colors.get(method), edgecolors="none")
            lo = np.nanmin([y_true.min(), y_pred.min()])
            hi = np.nanmax([y_true.max(), y_pred.max()])
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", lw=1)
            if r == 0:
                ax.set_title(scenario)
            if c == 0:
                ax.set_ylabel(method)
            if r == len(available) - 1:
                ax.set_xlabel("Observed")
    fig.savefig(output_file)
    plt.close(fig)

