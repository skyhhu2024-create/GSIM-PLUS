import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
STEP2_DIR = ROOT / "02_Feature_Table"
df = pd.read_csv(STEP2_DIR / "station_features_with_meteo.csv")

anchor = df[df["category"] == "anchor"]
target = df[df["category"] == "target"]

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="gray")
ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="lightgray")

ax.scatter(
    anchor["longitude"], anchor["latitude"],
    s=3, c="#4393c3", alpha=0.6, linewidths=0,
    transform=ccrs.PlateCarree(), label=f"Anchor (n={len(anchor)})", zorder=2,
)
ax.scatter(
    target["longitude"], target["latitude"],
    s=3, c="#d6604d", alpha=0.6, linewidths=0,
    transform=ccrs.PlateCarree(), label=f"Target (n={len(target)})", zorder=3,
)

ax.legend(loc="lower left", fontsize=11, frameon=True, markerscale=4,
          fancybox=False, edgecolor="gray")

plt.tight_layout()
out = STEP2_DIR / "global_anchor_target_distribution.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
