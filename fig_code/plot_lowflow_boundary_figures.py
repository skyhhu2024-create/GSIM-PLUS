from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(
    __import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])
).resolve()
PAPER_DIR = PROJECT_DIR / "111-paper"
ANALYSIS_DIR = PAPER_DIR / "lowflow_gap_boundary_analysis"
OUT_PNG = PAPER_DIR / "Fig11_why_002_threshold.png"
OUT_PDF = PAPER_DIR / "Fig11_why_002_threshold.pdf"

FLOW_ORDER = ["<0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05", ">0.05"]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    df = pd.read_csv(ANALYSIS_DIR / "focused_long_gap_flow_summary.csv")
    df = df[df["scenario"] == "cont_12m"].copy()
    df["flow_bin"] = pd.Categorical(df["flow_bin"], FLOW_ORDER, ordered=True)
    df = df.sort_values("flow_bin")

    x = np.arange(len(FLOW_ORDER))
    instability_pct = df["frac_nse_lt0"].to_numpy() * 100.0
    median_nse = df["median_nse"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=600, constrained_layout=True)

    bar_colors = ["#D55E00", "#E69F00", "#F0E442", "#56B4E9", "#0072B2"]
    ax.bar(x, instability_pct, color=bar_colors, width=0.68, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Stations with NSE < 0 (%)")
    ax.set_xlabel("Median flow bin (m$^3$ s$^{-1}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(FLOW_ORDER)
    ax.set_ylim(0, max(60, instability_pct.max() + 8))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, median_nse, color="black", marker="o", linewidth=2.0, markersize=5)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Median station-level NSE")
    ax2.spines["top"].set_visible(False)

    for xi, yi in zip(x, instability_pct):
        ax.text(xi, yi + 1.2, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)
    for xi, yi in zip(x, median_nse):
        ax2.text(xi, yi - 0.08 if yi > 0 else yi + 0.08, f"{yi:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Why a low-flow guard is needed")

    legend_handles = [
        plt.Line2D([0], [0], color="black", marker="o", linewidth=2, label="Median NSE"),
        plt.Rectangle((0, 0), 1, 1, color="#E69F00", ec="black", lw=0.6, label="Instability rate (NSE < 0)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
