import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import STEP1_DIR, STEP3_DIR, STEP4_DIR
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


def main():
    skip_plots = os.environ.get("GSIM_SKIP_PLOTS", "0") == "1"
    selected_methods = os.environ.get("GSIM_METHODS", "").strip() or None
    anchor_ids = pd.read_csv(STEP1_DIR / "anchor_stations.csv")["station_id"].tolist()
    target_ids = pd.read_csv(STEP1_DIR / "evaluable_target_stations.csv")["station_id"].tolist()
    similarity_df = pd.read_csv(STEP3_DIR / "top_5_similar_stations.csv")

    print("Loading data...")
    anchor_data = load_anchor_data(anchor_ids)
    target_base_data = load_target_base_data(target_ids)

    print("Training reusable models...")
    models = train_reusable_models(
        anchor_data,
        similarity_df,
        task_base_data=target_base_data,
        method_names=selected_methods,
    )

    print("Creating validation set...")
    validation_set = core.create_gap_rate_validation_set(target_base_data, 0.3)

    print("Running methods...")
    all_predictions, results, common_keys, method_names = run_all_methods(
        anchor_data,
        validation_set,
        similarity_df,
        models,
        method_names=selected_methods,
    )
    comparison_df = save_method_outputs(
        STEP4_DIR, "random_30pct", validation_set, all_predictions, results, common_keys, method_names
    )

    summary_df = comparison_df.copy()
    summary_df["scenario"] = "random_30pct"
    summary_df["method"] = summary_df["Method"]
    summary_df.to_csv(STEP4_DIR / "random_30pct_summary.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "scenario": "random_30pct",
                "gap_rate": 0.3,
                "n_validation_stations": len(validation_set),
                "n_common_points": len(common_keys),
            }
        ]
    )
    metadata.to_csv(STEP4_DIR / "scenario_metadata.csv", index=False)

    if not skip_plots:
        scenario_outputs = {
            "random_30pct": {
                "predictions": dict(zip(method_names, all_predictions)),
                "common_keys": common_keys,
            }
        }
        plot_scatter_panel(scenario_outputs, ["random_30pct"], STEP4_DIR / "scatter_panel_random_30pct.png")

    with open(STEP4_DIR / "random_30pct_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": "random_30pct",
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
    print(f"Saved to: {STEP4_DIR}")


if __name__ == "__main__":
    main()
