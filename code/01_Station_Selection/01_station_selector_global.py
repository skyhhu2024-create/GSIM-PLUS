import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import (
    ANCHOR_THRESHOLD,
    EVALUABLE_MIN_MONTHS,
    MONTHLY_DIR,
    N_STUDY_MONTHS,
    STEP1_DIR,
    STUDY_END_YEAR,
    STUDY_START_YEAR,
    TARGET_MAX,
    TARGET_MIN,
)
from gsim_plus_utils import compute_valid_months, extract_gap_lengths, read_station_study_period


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def analyze_station(file_path):
    station_id = file_path.stem
    study_df = read_station_study_period(station_id)

    full_index = pd.date_range(f"{STUDY_START_YEAR}-01-31", f"{STUDY_END_YEAR}-12-31", freq="ME")
    station = pd.DataFrame(index=full_index)
    station = station.join(study_df.set_index("date"), how="left")

    valid_mask = station["MEAN"].notna().values if "MEAN" in station.columns else station["n.available"].fillna(0).gt(0).values
    valid_months = int(valid_mask.sum())
    completeness = valid_months / N_STUDY_MONTHS
    gap_lengths = extract_gap_lengths(~valid_mask)
    max_gap = max(gap_lengths) if gap_lengths else 0

    if completeness >= ANCHOR_THRESHOLD:
        category = "anchor"
    elif TARGET_MIN <= completeness < TARGET_MAX:
        category = "target"
    else:
        category = "insufficient"

    evaluable = category == "target" and valid_months >= EVALUABLE_MIN_MONTHS

    return {
        "station_id": station_id,
        "region_code": station_id.split("_")[0] if "_" in station_id else station_id[:2],
        "total_months": N_STUDY_MONTHS,
        "valid_months": valid_months,
        "missing_months": N_STUDY_MONTHS - valid_months,
        "completeness": round(completeness, 4),
        "n_gaps": len(gap_lengths),
        "mean_gap_length": round(float(pd.Series(gap_lengths).mean()) if gap_lengths else 0.0, 4),
        "max_gap_length": int(max_gap),
        "category": category,
        "evaluable_target": bool(evaluable),
        "gap_lengths": " ".join(map(str, gap_lengths)),
    }


def plot_selection_summary(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    counts = df["category"].value_counts().reindex(["anchor", "target", "insufficient"], fill_value=0)
    axes[0].bar(counts.index, counts.values, color=["#2a9d8f", "#457b9d", "#c8553d"])
    axes[0].set_title("Station Categories")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].hist(df["completeness"], bins=40, color="#457b9d", edgecolor="white")
    axes[1].axvline(ANCHOR_THRESHOLD, color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(TARGET_MIN, color="orange", linestyle="--", linewidth=1.5)
    axes[1].set_title("Completeness Distribution")
    axes[1].set_xlabel("Completeness")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    evaluable_counts = pd.Series(
        {
            "evaluable_target": int(df["evaluable_target"].sum()),
            "other_target": int(((df["category"] == "target") & (~df["evaluable_target"])).sum()),
        }
    )
    axes[2].bar(evaluable_counts.index, evaluable_counts.values, color=["#2a9d8f", "#8d99ae"])
    axes[2].set_title("Target Usability")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")

    fig.savefig(STEP1_DIR / "station_selection_summary.png")
    plt.close(fig)


def main():
    files = sorted(MONTHLY_DIR.glob("*.mon"))
    results = []
    print(f"Scanning {len(files)} station files...")
    for i, file_path in enumerate(files, start=1):
        if i % 2000 == 0:
            print(f"  Processed {i}/{len(files)}...")
        try:
            results.append(analyze_station(file_path))
        except Exception:
            continue

    df = pd.DataFrame(results).sort_values("station_id")
    df.to_csv(STEP1_DIR / "all_stations_1995_2015.csv", index=False)

    anchor_df = df[df["category"] == "anchor"].copy()
    target_df = df[df["category"] == "target"].copy()
    evaluable_df = df[df["evaluable_target"]].copy()

    anchor_df.to_csv(STEP1_DIR / "anchor_stations.csv", index=False)
    target_df.to_csv(STEP1_DIR / "target_stations.csv", index=False)
    evaluable_df.to_csv(STEP1_DIR / "evaluable_target_stations.csv", index=False)
    evaluable_df[["station_id"]].to_csv(STEP1_DIR / "evaluable_target_names.csv", index=False)

    summary = {
        "n_all": int(len(df)),
        "n_anchor": int(len(anchor_df)),
        "n_target": int(len(target_df)),
        "n_insufficient": int((df["category"] == "insufficient").sum()),
        "n_evaluable_target": int(len(evaluable_df)),
        "anchor_threshold": ANCHOR_THRESHOLD,
        "target_min": TARGET_MIN,
        "target_max": TARGET_MAX,
        "evaluable_min_months": EVALUABLE_MIN_MONTHS,
    }
    with open(STEP1_DIR / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_selection_summary(df)

    print("Selection summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {STEP1_DIR}")


if __name__ == "__main__":
    main()
