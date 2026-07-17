"""
Extract observed vs filled data for 4 key GRDC stations
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
EXTERNAL = ROOT / "external_data" / "GRDC"
EU_GRDC_DIR = os.environ.get("GSIM_PLUS_EU_GRDC_DIR", str(EXTERNAL / "Europe" / "parsed_csv"))
USA_GRDC_DIR = os.environ.get("GSIM_PLUS_USA_GRDC_DIR", str(EXTERNAL / "USA" / "parsed_csv"))
GSIM_TARGET = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
GSIM_ANCHOR = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"

# Target stations
STATIONS = {
    "FR_0001112": {"grdc_id": "6135110", "grdc_dir": EU_GRDC_DIR},
    "US_0005774": {"grdc_id": "4125804", "grdc_dir": USA_GRDC_DIR},
    "US_0002812": {"grdc_id": "4123245", "grdc_dir": USA_GRDC_DIR},
    "US_0004183": {"grdc_id": "4119313", "grdc_dir": USA_GRDC_DIR},
}

def load_grdc(grdc_path):
    grdc = pd.read_csv(grdc_path)
    grdc["date"] = pd.to_datetime(grdc["data"])
    grdc = grdc[["date", "MEAN"]].dropna(subset=["MEAN"])
    grdc = grdc[(grdc["date"] >= "1995-01-01") & (grdc["date"] <= "2015-12-31")]
    grdc["ym"] = grdc["date"].dt.to_period("M")
    return grdc

def load_gsim(gsim_id):
    # Try target first, then anchor
    for directory in [GSIM_TARGET, GSIM_ANCHOR]:
        path = directory / f"{gsim_id}.csv"
        if path.exists():
            gsim = pd.read_csv(path)
            gsim["date"] = pd.to_datetime(gsim["date"])
            gsim["ym"] = gsim["date"].dt.to_period("M")
            return gsim
    return None

def compute_metrics(gsim_v, grdc_v):
    n = len(gsim_v)
    if n == 0:
        return {}
    mae = np.mean(np.abs(gsim_v - grdc_v))
    rmse = np.sqrt(np.mean((gsim_v - grdc_v) ** 2))
    r = np.corrcoef(gsim_v, grdc_v)[0, 1] if n > 1 else np.nan
    ss_res = np.sum((gsim_v - grdc_v) ** 2)
    ss_tot = np.sum((grdc_v - np.mean(grdc_v)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": n, "R": r, "NSE": nse, "RMSE": rmse, "MAE": mae}

out_dir = ROOT / "09 GRDC交叉验证" / "4stations_comparison"
out_dir.mkdir(exist_ok=True)

for gsim_id, info in STATIONS.items():
    grdc_id = info["grdc_id"]
    grdc_dir = Path(info["grdc_dir"])

    grdc_path = grdc_dir / f"{grdc_id}.csv"
    if not grdc_path.exists():
        print(f"GRDC file not found: {grdc_path}")
        continue

    grdc = load_grdc(grdc_path)
    gsim = load_gsim(gsim_id)

    if gsim is None:
        print(f"GSIM file not found: {gsim_id}")
        continue

    merged = gsim.merge(grdc[["ym", "MEAN"]], on="ym", how="inner")
    merged = merged.rename(columns={"MEAN": "grdc_observed", "final_streamflow": "gsim_filled"})

    fill_mask = merged["fill_method"] != "OBSERVED"

    # Compute metrics
    all_m = compute_metrics(merged["gsim_filled"].values, merged["grdc_observed"].values)
    fill_m = compute_metrics(
        merged.loc[fill_mask, "gsim_filled"].values,
        merged.loc[fill_mask, "grdc_observed"].values
    )

    # Save comparison data
    output = merged[["date", "year", "month", "grdc_observed", "gsim_filled", "fill_method", "quality_flag"]]
    output.to_csv(out_dir / f"{gsim_id}_{grdc_id}_comparison.csv", index=False)

    print(f"\n{gsim_id} (GRDC {grdc_id}):")
    print(f"  Total months: {all_m['n']}")
    print(f"  Filled months: {fill_m['n']}")
    print(f"  Fill NSE: {fill_m['NSE']:.3f}")
    print(f"  Fill R: {fill_m['R']:.3f}")
    print(f"  Fill RMSE: {fill_m['RMSE']:.1f}")
    print(f"  Fill MAE: {fill_m['MAE']:.1f}")
    print(f"  Saved to: {gsim_id}_{grdc_id}_comparison.csv")

print(f"\nAll comparison files saved to: {out_dir}")


