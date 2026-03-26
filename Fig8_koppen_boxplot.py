"""
Figure 8 — Köppen Climate Zone Performance Analysis
  Fig8a: Boxplot of per-station NSE across Köppen major zones (8 methods)
  Fig8b: Boxplot of per-station KGE across Köppen major zones (8 methods)
  Fig8c: Boxplot of per-station MAE across Köppen major zones (8 methods)
  Fig8d: Boxplot of per-station Bias across Köppen major zones (8 methods)
  Fig8e: Boxplot of per-station RMSE across Köppen major zones (8 methods)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
PRED_DIR = ROOT / "04_Random_30pct_Validation" / "random_30pct"
FEAT_CSV = ROOT / "02_Feature_Table" / "station_features_with_meteo.csv"
OUTDIR = ROOT / "111-paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

# ── config ───────────────────────────────────────────────────────────────
METHODS = ["MAML", "MAML_DonorTrend", "RandomForest", "Linear",
           "LSTM", "KNN", "SeasonalMean", "IDW"]
METHOD_LABELS = {
    "MAML":          "MAML",
    "MAML_DonorTrend":"MAML DonorTrend",
    "RandomForest":  "Random Forest",
    "Linear":        "Linear",
    "LSTM":          "LSTM",
    "KNN":           "KNN",
    "SeasonalMean":  "Seasonal Mean",
    "IDW":           "IDW",
}
NPG_COLORS = {
    "MAML":          "#E64B35",
    "MAML_DonorTrend":"#00A087",
    "RandomForest":  "#3C5488",
    "Linear":        "#4DBBD5",
    "LSTM":          "#F39B7F",
    "KNN":           "#8491B4",
    "SeasonalMean":  "#91D1C2",
    "IDW":           "#B09C85",
}
KOPPEN_ORDER = ["A", "B", "C", "D", "E"]
KOPPEN_LABELS = {
    "A": "A (Tropical)",
    "B": "B (Arid)",
    "C": "C (Temperate)",
    "D": "D (Continental)",
    "E": "E (Polar)",
}

# ── load station features (for Köppen zone) ──────────────────────────────
feat = pd.read_csv(FEAT_CSV)
station_koppen = feat.set_index("station_id")["kg_major"].to_dict()

# ── compute per-station NSE & KGE for each method ───────────────────────
def calc_nse(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def calc_kge(true, pred):
    if np.std(true) == 0 or np.std(pred) == 0:
        return np.nan
    r = np.corrcoef(true, pred)[0, 1]
    alpha = np.std(pred) / np.std(true)
    beta = np.mean(pred) / np.mean(true) if np.mean(true) != 0 else np.nan
    if np.isnan(r) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

all_results = []
for method in METHODS:
    print(f"  Computing per-station metrics: {method}")
    pred_file = PRED_DIR / f"{method}_predictions.csv"
    df = pd.read_csv(pred_file)

    for station, grp in df.groupby("target_station"):
        if len(grp) < 6:  # skip stations with too few points
            continue
        true = grp["true"].values
        pred = grp["pred"].values
        nse = calc_nse(true, pred)
        kge = calc_kge(true, pred)
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        bias = np.mean(pred - true) / np.mean(true) if np.mean(true) != 0 else np.nan
        kg = station_koppen.get(station, None)
        if kg is None or kg not in KOPPEN_ORDER:
            continue
        all_results.append({
            "station": station,
            "method": method,
            "NSE": nse,
            "KGE": kge,
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "koppen": kg,
        })

results = pd.DataFrame(all_results)
print(f"Total station-method pairs: {len(results)}")

# ── all 8 methods for boxplot ────────────────────────────────────────────
PLOT_METHODS = METHODS

# ── Fig8a-e: Grouped boxplot per Köppen zone ─────────────────────────────
METRIC_CONFIG = [
    ("NSE",  "a", -1.0, 1.05,  True),   # metric, suffix, ymin, ymax, show_zero_line
    ("KGE",  "b", -0.8, 1.05,  True),
    ("MAE",  "c",  None, None, False),
    ("Bias", "d", -1.0, 1.0,   True),
    ("RMSE", "e",  None, None, False),
]

for metric, suffix, ymin, ymax, zero_line in METRIC_CONFIG:
    fig, ax = plt.subplots(figsize=(180 / 25.4, 80 / 25.4), dpi=300)

    n_zones = len(KOPPEN_ORDER)
    n_methods = len(PLOT_METHODS)
    box_width = 0.10
    positions_all = []
    colors_all = []
    data_all = []
    labels_placed = set()

    for zi, zone in enumerate(KOPPEN_ORDER):
        zone_data = results[results["koppen"] == zone]
        for mi, method in enumerate(PLOT_METHODS):
            mdata = zone_data[zone_data["method"] == method][metric].dropna()
            pos = zi * (n_methods + 1) * box_width + mi * box_width
            positions_all.append(pos)
            colors_all.append(NPG_COLORS[method])
            data_all.append(mdata.values)

    bp = ax.boxplot(
        data_all,
        positions=positions_all,
        widths=box_width * 0.85,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
        boxprops=dict(linewidth=0.4),
    )

    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # x-axis: zone labels at group centers
    group_centers = []
    for zi in range(n_zones):
        center = zi * (n_methods + 1) * box_width + (n_methods - 1) * box_width / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([KOPPEN_LABELS[z] for z in KOPPEN_ORDER], fontsize=7)

    # set y-axis limits
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    ylabel = metric if metric != "Bias" else "Relative Bias"
    if metric == "MAE":
        ylabel = r"MAE (m$^3$/s)"
    elif metric == "RMSE":
        ylabel = r"RMSE (m$^3$/s)"
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # reference line
    if zero_line:
        ax.axhline(0, color="grey", ls=":", lw=0.4, zorder=0)

    # count annotations (below x-axis labels)
    ybot = ax.get_ylim()[0]
    for zi, zone in enumerate(KOPPEN_ORDER):
        n_stations = results[(results["koppen"] == zone) & (results["method"] == PLOT_METHODS[0])].shape[0]
        ax.text(group_centers[zi], ybot + 0.03,
                f"n={n_stations}", ha="center", va="bottom", fontsize=5.5, color="#666666")

    # legend
    legend_elements = [Patch(facecolor=NPG_COLORS[m], alpha=0.75,
                             label=METHOD_LABELS[m]) for m in PLOT_METHODS]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=5.5,
              handlelength=0.8, handletextpad=0.3, ncol=4, columnspacing=0.6)

    fig.tight_layout()

    fname = f"Fig8{suffix}_koppen_{metric}"
    fig.savefig(str(OUTDIR / f"{fname}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUTDIR / f"{fname}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fname}")

# ── Print summary table ──────────────────────────────────────────────────
print("\n=== Per-zone median NSE (MAML) ===")
maml = results[results["method"] == "MAML"]
for zone in KOPPEN_ORDER:
    zd = maml[maml["koppen"] == zone]
    print(f"  {KOPPEN_LABELS[zone]}: median NSE={zd['NSE'].median():.4f}, "
          f"median KGE={zd['KGE'].median():.4f}, n={len(zd)}")

print("\n=== Per-zone median NSE (MAML_DonorTrend) ===")
maml_dt = results[results["method"] == "MAML_DonorTrend"]
for zone in KOPPEN_ORDER:
    zd = maml_dt[maml_dt["koppen"] == zone]
    print(f"  {KOPPEN_LABELS[zone]}: median NSE={zd['NSE'].median():.4f}, "
          f"median KGE={zd['KGE'].median():.4f}, n={len(zd)}")
