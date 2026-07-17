import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import H1_WEIGHTS, H2_WEIGHTS, H3_WEIGHTS, STEP1_DIR, STEP3_DIR, STEP6_DIR
from gsim_plus_utils import gap_bin, parse_gap_lengths, weighted_choice_from_bins
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


SCENARIOS = {
    "H1_sparse_dominant": H1_WEIGHTS,
    "H2_balanced_mixed": H2_WEIGHTS,
    "H3_long_gap_dominant": H3_WEIGHTS,
}
TARGET_GAP = 0.30


def build_gap_bin_pools():
    target_df = pd.read_csv(STEP1_DIR / "target_stations.csv")
    pools = defaultdict(list)
    for text in target_df["gap_lengths"]:
        for length in parse_gap_lengths(text):
            pools[gap_bin(length)].append(length)
    return {k: np.array(v, dtype=int) for k, v in pools.items() if len(v) > 0}


def find_run_starts(mask, seg_len):
    starts = []
    for start in range(len(mask) - seg_len + 1):
        if mask[start : start + seg_len].all():
            starts.append(start)
    return starts


def create_hybrid_validation_set(target_base_data, bin_pools, scenario_weights):
    validation_set = {}
    mask_rows = []
    for station_id, data in target_base_data.items():
        valid_mask = data["valid"].copy()
        valid_count = int(valid_mask.sum())
        if valid_count < 12:
            continue
        n_hide = max(1, int(valid_count * TARGET_GAP))
        if valid_count - n_hide < 12:
            continue

        rng = np.random.default_rng(core.station_seed(station_id, int(TARGET_GAP * 100)))
        available_mask = valid_mask.copy()
        hidden = set()

        while len(hidden) < n_hide:
            remaining = n_hide - len(hidden)
            chosen_bin = weighted_choice_from_bins(rng, scenario_weights)
            pool = bin_pools.get(chosen_bin, np.array([1], dtype=int))
            feasible = pool[pool <= remaining]
            if len(feasible) == 0:
                seg_len = 1
            else:
                seg_len = int(rng.choice(feasible))

            starts = find_run_starts(available_mask, seg_len)
            if not starts:
                seg_len = 1
                starts = find_run_starts(available_mask, seg_len)
                if not starts:
                    break
            start_idx = int(rng.choice(starts))
            for idx in range(start_idx, start_idx + seg_len):
                hidden.add(int(idx))
                available_mask[idx] = False

        hide_indices = np.array(sorted(hidden), dtype=int)
        entry = core.build_validation_entry(station_id, data, hide_indices)
        if entry is not None:
            validation_set[station_id] = entry
            mask_rows.append({"station_id": station_id, "hidden_count": len(hide_indices)})

    return validation_set, pd.DataFrame(mask_rows)


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
    bin_pools = build_gap_bin_pools()

    scenario_outputs = {}
    summary_rows = []
    metadata_rows = []
    for scenario_name, weights in SCENARIOS.items():
        print(f"Running {scenario_name}...")
        validation_set, mask_df = create_hybrid_validation_set(target_base_data, bin_pools, weights)
        all_predictions, results, common_keys, method_names = run_all_methods(
            anchor_data,
            validation_set,
            similarity_df,
            models,
            method_names=selected_methods,
        )
        comparison_df = save_method_outputs(
            STEP6_DIR, scenario_name, validation_set, all_predictions, results, common_keys, method_names
        )
        mask_df.to_csv(STEP6_DIR / scenario_name / "mask_metadata.csv", index=False)

        metadata_rows.append(
            {
                "scenario": scenario_name,
                "n_validation_stations": len(validation_set),
                "n_common_points": len(common_keys),
                "target_gap": TARGET_GAP,
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
    summary_df.to_csv(STEP6_DIR / "hybrid_h123_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(STEP6_DIR / "scenario_metadata.csv", index=False)
    if not skip_plots:
        plot_summary_heatmap(summary_df, "scenario", STEP6_DIR / "hybrid_h123_heatmap.png", method_names=selected_methods)
        plot_scatter_panel(scenario_outputs, list(SCENARIOS.keys()), STEP6_DIR / "hybrid_h123_scatter.png")

    with open(STEP6_DIR / "hybrid_h123_results.json", "w", encoding="utf-8") as f:
        json.dump({"scenarios": metadata_rows}, f, indent=2)

    if skip_plots:
        print("Plots skipped because GSIM_SKIP_PLOTS=1")
    print(f"Saved to: {STEP6_DIR}")


if __name__ == "__main__":
    main()
