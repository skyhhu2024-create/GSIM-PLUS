"""
Export Köppen climate zone statistics to CSV
Based on Fig8_koppen_boxplot.py logic
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
PRED_DIR = ROOT / "04_Random_30pct_Validation" / "random_30pct"
FEAT_CSV = ROOT / "02_Feature_Table" / "station_features_with_meteo.csv"
OUTDIR = ROOT / "111-paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Config
METHODS = ["DTRR", "MAML", "RandomForest", "Linear", "LSTM", "KNN", "SeasonalMean", "IDW"]
METHOD_LABELS = {
    "MAML": "MAML",
    "DTRR": "DTRR",
    "RandomForest": "RF",
    "Linear": "Linear",
    "LSTM": "LSTM",
    "KNN": "KNN",
    "SeasonalMean": "Seasonal Mean",
    "IDW": "IDW",
}
KOPPEN_ORDER = ["A", "B", "C", "D", "E"]
KOPPEN_LABELS = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Continental",
    "E": "Polar",
}

# Load station features
feat = pd.read_csv(FEAT_CSV)
station_koppen = feat.set_index("station_id")["kg_major"].to_dict()

def calc_metrics(true, pred):
    """Calculate NSE, KGE, Bias"""
    # NSE
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # KGE
    r = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else np.nan
    alpha = np.std(pred) / np.std(true) if np.std(true) > 0 else np.nan
    beta = np.mean(pred) / np.mean(true) if np.mean(true) > 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Bias (%)
    bias = np.mean((pred - true) / np.where(true == 0, 1e-6, true)) * 100

    return nse, kge, bias

print("Loading prediction data...")
all_results = []
for method in METHODS:
    print(f"  Processing: {method}")
    pred_file = PRED_DIR / f"{method}_predictions.csv"
    if not pred_file.exists():
        print(f"    Warning: {pred_file} not found, skipping")
        continue

    df = pd.read_csv(pred_file)
    for station, grp in df.groupby("target_station"):
        if len(grp) < 6:
            continue

        nse, kge, bias = calc_metrics(grp["true"].values, grp["pred"].values)
        kg = station_koppen.get(station, None)

        if kg in KOPPEN_ORDER and np.isfinite(nse) and np.isfinite(kge):
            all_results.append({
                "method": method,
                "station": station,
                "koppen": kg,
                "NSE": nse,
                "KGE": kge,
                "Bias": bias
            })

results = pd.DataFrame(all_results)
print(f"\nTotal records: {len(results)}")

# Export 1: Station-level data (all records)
output_file = OUTDIR / "koppen_climate_zone_statistics.csv"
results.to_csv(output_file, index=False)
print(f"\nExported station-level data: {output_file}")

# Export 2: Summary statistics by method and climate zone
summary_list = []
for method in METHODS:
    for koppen in KOPPEN_ORDER:
        subset = results[(results["method"] == method) & (results["koppen"] == koppen)]
        if len(subset) > 0:
            summary_list.append({
                "method": METHOD_LABELS[method],
                "climate_zone": KOPPEN_LABELS[koppen],
                "climate_code": koppen,
                "n_stations": len(subset),
                "NSE_mean": subset["NSE"].mean(),
                "NSE_median": subset["NSE"].median(),
                "NSE_std": subset["NSE"].std(),
                "NSE_q25": subset["NSE"].quantile(0.25),
                "NSE_q75": subset["NSE"].quantile(0.75),
                "KGE_mean": subset["KGE"].mean(),
                "KGE_median": subset["KGE"].median(),
                "KGE_std": subset["KGE"].std(),
                "KGE_q25": subset["KGE"].quantile(0.25),
                "KGE_q75": subset["KGE"].quantile(0.75),
                "Bias_mean": subset["Bias"].mean(),
                "Bias_median": subset["Bias"].median(),
                "Bias_std": subset["Bias"].std(),
            })

summary_df = pd.DataFrame(summary_list)
summary_file = OUTDIR / "koppen_climate_zone_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Exported summary statistics: {summary_file}")

# Export 3: Pivot table for easy comparison
pivot_nse = results.pivot_table(
    values="NSE",
    index="method",
    columns="koppen",
    aggfunc="median"
)
pivot_nse.to_csv(OUTDIR / "koppen_NSE_pivot.csv")

pivot_kge = results.pivot_table(
    values="KGE",
    index="method",
    columns="koppen",
    aggfunc="median"
)
pivot_kge.to_csv(OUTDIR / "koppen_KGE_pivot.csv")

pivot_bias = results.pivot_table(
    values="Bias",
    index="method",
    columns="koppen",
    aggfunc="median"
)
pivot_bias.to_csv(OUTDIR / "koppen_Bias_pivot.csv")

print("Exported pivot tables: koppen_NSE_pivot.csv, koppen_KGE_pivot.csv, koppen_Bias_pivot.csv")
print("\nDone!")


