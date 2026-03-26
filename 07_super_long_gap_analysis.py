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

from gsim_plus_config import STEP1_DIR, STEP3_DIR, STEP7_DIR, SUPER_LONG_MIN_MONTHS
from gsim_plus_utils import parse_gap_lengths
from validation_wrappers import (
    METHOD_ORDER,
    core,
    load_anchor_data,
    load_target_base_data,
    plot_scatter_panel,
    run_all_methods,
    save_method_outputs,
    train_reusable_models,
)


def build_super_long_pool():
    target_df = pd.read_csv(STEP1_DIR / "target_stations.csv")
    pool = []
    for text in target_df["gap_lengths"]:
        for value in parse_gap_lengths(text):
            if value >= SUPER_LONG_MIN_MONTHS:
                pool.append(value)
    if not pool:
        pool = [SUPER_LONG_MIN_MONTHS]
    return np.array(pool, dtype=int)


def create_super_long_validation_set(target_base_data, super_long_pool):
    validation_set = {}
    rows = []
    for station_id, data in target_base_data.items():
        valid_mask = data["valid"]
        valid_count = int(valid_mask.sum())
        if valid_count < 120:
            continue

        rng = np.random.default_rng(core.station_seed(station_id, SUPER_LONG_MIN_MONTHS))
        feasible = super_long_pool[super_long_pool <= max(24, valid_count - 12)]
        if len(feasible) == 0:
            continue
        seg_len = int(rng.choice(feasible))
        starts = []
        for start_idx in range(len(valid_mask) - seg_len + 1):
            if valid_mask[start_idx : start_idx + seg_len].all():
                starts.append(start_idx)
        if not starts:
            continue

        start_idx = int(rng.choice(starts))
        hide_indices = list(range(start_idx, start_idx + seg_len))
        entry = core.build_validation_entry(station_id, data, hide_indices)
        if entry is not None:
            validation_set[station_id] = entry
            rows.append({"station_id": station_id, "gap_length": seg_len, "start_idx": start_idx})
    return validation_set, pd.DataFrame(rows)


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
    super_long_pool = build_super_long_pool()

    validation_set, mask_df = create_super_long_validation_set(target_base_data, super_long_pool)
    all_predictions, results, common_keys, method_names = run_all_methods(
        anchor_data,
        validation_set,
        similarity_df,
        models,
        method_names=selected_methods,
    )
    comparison_df = save_method_outputs(
        STEP7_DIR, "super_long_25plus", validation_set, all_predictions, results, common_keys, method_names
    )
    mask_df.to_csv(STEP7_DIR / "super_long_25plus" / "mask_metadata.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "scenario": "super_long_25plus",
                "n_validation_stations": len(validation_set),
                "n_common_points": len(common_keys),
                "mean_gap_length": mask_df["gap_length"].mean() if not mask_df.empty else 0,
                "median_gap_length": mask_df["gap_length"].median() if not mask_df.empty else 0,
            }
        ]
    )
    metadata.to_csv(STEP7_DIR / "scenario_metadata.csv", index=False)
    comparison_df.to_csv(STEP7_DIR / "super_long_summary.csv", index=False)
    if not skip_plots:
        plot_scatter_panel(
            {"super_long_25plus": {"predictions": dict(zip(method_names, all_predictions)), "common_keys": common_keys}},
            ["super_long_25plus"],
            STEP7_DIR / "super_long_scatter.png",
        )

    with open(STEP7_DIR / "super_long_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": "super_long_25plus",
                "n_validation_stations": len(validation_set),
                "n_common_points": len(common_keys),
                "methods": method_names,
                "results": results,
            },
            f,
            indent=2,
        )

    print(comparison_df.to_string(index=False))
    if skip_plots:
        print("Plots skipped because GSIM_SKIP_PLOTS=1")
    print(f"Saved to: {STEP7_DIR}")


if __name__ == "__main__":
    main()
