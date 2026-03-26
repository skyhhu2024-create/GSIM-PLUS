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

from gsim_plus_config import STEP1_DIR, STEP3_DIR, STEP5_DIR
from validation_wrappers import (
    METHOD_ORDER,
    core,
    load_anchor_data,
    load_target_base_data,
    plot_scatter_panel,
    plot_summary_heatmap,
    run_all_methods,
    save_method_outputs,
    train_reusable_models,
)


SCENARIOS = {"3_months": 3, "6_months": 6, "12_months": 12}


def create_continuous_validation_set(target_base_data, gap_length):
    validation_set = {}
    for station_id, data in target_base_data.items():
        valid_mask = data["valid"]
        valid_indices = valid_mask.nonzero()[0]
        if len(valid_indices) < 12:
            continue

        available_starts = []
        for start_idx in range(len(valid_mask) - gap_length + 1):
            if valid_mask[start_idx : start_idx + gap_length].all():
                available_starts.append(start_idx)
        if not available_starts:
            continue

        rng = np.random.default_rng(core.station_seed(station_id, gap_length))
        start_idx = int(rng.choice(available_starts))
        hide_indices = list(range(start_idx, start_idx + gap_length))

        entry = core.build_validation_entry(station_id, data, hide_indices)
        if entry is not None:
            validation_set[station_id] = entry
    return validation_set


def main():
    skip_plots = os.environ.get("GSIM_SKIP_PLOTS", "0") == "1"
    selected_methods = os.environ.get("GSIM_METHODS", "").strip() or None
    anchor_ids = pd.read_csv(STEP1_DIR / "anchor_stations.csv")["station_id"].tolist()
    target_ids = pd.read_csv(STEP1_DIR / "evaluable_target_stations.csv")["station_id"].tolist()
    similarity_df = pd.read_csv(STEP3_DIR / "top_5_similar_stations.csv")

    anchor_data = load_anchor_data(anchor_ids)
    target_base_data = load_target_base_data(target_ids)
    models = train_reusable_models(
        anchor_data,
        similarity_df,
        task_base_data=target_base_data,
        method_names=selected_methods,
    )

    scenario_outputs = {}
    summary_rows = []
    metadata_rows = []
    for scenario_name, gap_length in SCENARIOS.items():
        print(f"Running {scenario_name}...")
        validation_set = create_continuous_validation_set(target_base_data, gap_length)
        all_predictions, results, common_keys, method_names = run_all_methods(
            anchor_data,
            validation_set,
            similarity_df,
            models,
            method_names=selected_methods,
        )
        comparison_df = save_method_outputs(
            STEP5_DIR, scenario_name, validation_set, all_predictions, results, common_keys, method_names
        )
        metadata_rows.append(
            {
                "scenario": scenario_name,
                "gap_length": gap_length,
                "n_validation_stations": len(validation_set),
                "n_common_points": len(common_keys),
            }
        )
        for _, row in comparison_df.iterrows():
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "method": row["Method"],
                    "NSE": row["NSE"],
                    "KGE": row["KGE"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "Bias": row["Bias"],
                    "n": row["n"],
                }
            )
        scenario_outputs[scenario_name] = {
            "predictions": dict(zip(method_names, all_predictions)),
            "common_keys": common_keys,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(STEP5_DIR / "continuous_gap_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(STEP5_DIR / "scenario_metadata.csv", index=False)
    if not skip_plots:
        plot_summary_heatmap(summary_df, "scenario", STEP5_DIR / "continuous_gap_heatmap.png", method_names=selected_methods)
        plot_scatter_panel(scenario_outputs, list(SCENARIOS.keys()), STEP5_DIR / "continuous_gap_scatter.png")

    with open(STEP5_DIR / "continuous_gap_results.json", "w", encoding="utf-8") as f:
        json.dump({"scenarios": metadata_rows}, f, indent=2)

    if skip_plots:
        print("Plots skipped because GSIM_SKIP_PLOTS=1")
    print(f"Saved to: {STEP5_DIR}")


if __name__ == "__main__":
    main()
