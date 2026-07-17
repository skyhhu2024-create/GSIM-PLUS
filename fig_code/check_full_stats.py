import pandas as pd
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
TARGET_DIR = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
ANCHOR_DIR = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"

dfs = []
for src_dir in [TARGET_DIR, ANCHOR_DIR]:
    for fp in src_dir.glob("*.csv"):
        dfs.append(pd.read_csv(fp))
df = pd.concat(dfs, ignore_index=True)

print(f"Total stations: {df['station_id'].nunique()}")
print(f"Total records: {len(df):,}")

qf = df["quality_flag"].value_counts()
total = len(df)
for q in ["Q0", "Q1", "Q2", "Q3"]:
    pct = qf.get(q, 0) / total * 100
    print(f"{q}: {qf.get(q, 0):,} ({pct:.1f}%)")

filled = df[df["fill_method"] != "OBSERVED"]
print(f"\nFilled points: {len(filled):,}")

per_station = df.groupby("station_id").agg(
    observed=("fill_method", lambda x: (x == "OBSERVED").sum()),
    total=("date", "count")
).reset_index()
per_station["pct_before"] = per_station["observed"] / 252 * 100
per_station["pct_after"] = per_station["total"] / 252 * 100

print(f"\nBefore: median={per_station['pct_before'].median():.1f}%, IQR={per_station['pct_before'].quantile(0.25):.1f}-{per_station['pct_before'].quantile(0.75):.1f}%")
print(f"After: median={per_station['pct_after'].median():.1f}%, IQR={per_station['pct_after'].quantile(0.25):.1f}-{per_station['pct_after'].quantile(0.75):.1f}%")
print(f"100% complete: {(per_station['pct_after'] == 100).sum()} ({(per_station['pct_after'] == 100).sum()/len(per_station)*100:.1f}%)")


