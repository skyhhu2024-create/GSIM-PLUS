"""
分析 GSIM 数据分布特征 + GRDC 验证失败原因诊断
"""
import os, sys, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# ── 路径 ──
ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
GSIM_DIR = os.environ.get("GSIM_MONTHLY_DIR", str(ROOT / "external_data" / "GSIM_monthly"))
GRDC_CSV = os.environ.get(
    "GSIM_PLUS_GRDC_SUMMARY",
    str(ROOT / "09 GRDC_CrossValidation" / "combined_validation_summary.csv"),
)
STEP01_DIR = os.environ.get("GSIM_PLUS_STEP01_DIR", str(ROOT / "01_Station_Selection"))

# ── 1. 快速扫描 GSIM 站点 1995-2015 数据完整度 ──
print("=" * 70)
print("1. GSIM 全球站点 1995-2015 数据分布扫描")
print("=" * 70)

files = sorted(glob.glob(os.path.join(GSIM_DIR, "*.csv")))
print(f"总文件数: {len(files)}")

# 随机采样分析（全扫太慢）
np.random.seed(42)
sample_idx = np.random.choice(len(files), min(3000, len(files)), replace=False)
sample_files = [files[i] for i in sorted(sample_idx)]

records = []
for f in sample_files:
    sid = os.path.basename(f).replace(".csv", "")
    country = sid[:2]
    try:
        lines = open(f, "r", encoding="utf-8", errors="ignore").readlines()
        # 解析 metadata
        lat = lon = area = np.nan
        for line in lines[:25]:
            if "latitude" in line:
                m = re.search(r":\s*([-\d.]+)", line)
                if m: lat = float(m.group(1))
            elif "longitude" in line:
                m = re.search(r":\s*([-\d.]+)", line)
                if m: lon = float(m.group(1))
            elif "area" in line and "km2" in line:
                m = re.search(r":\s*([-\d.]+)", line)
                if m: area = float(m.group(1))

        # 找数据起始行
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('"') and "date" not in line.lower() and re.match(r'"?\d{4}', line.strip().strip('"')):
                data_start = i
                break

        if data_start is None:
            continue

        # 解析月数据
        valid_months_95_15 = 0
        total_months_95_15 = 0
        all_values = []
        zero_count = 0

        for line in lines[data_start:]:
            line = line.strip().strip('"')
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 2:
                continue
            date_str = parts[0].strip()
            try:
                year = int(date_str[:4])
            except:
                continue
            if 1995 <= year <= 2015:
                total_months_95_15 += 1
                mean_val = parts[1].strip() if len(parts) > 1 else "NA"
                if mean_val not in ("NA", "", "nan"):
                    try:
                        v = float(mean_val)
                        valid_months_95_15 += 1
                        all_values.append(v)
                        if v == 0:
                            zero_count += 1
                    except:
                        pass

        if total_months_95_15 == 0:
            continue

        completeness = valid_months_95_15 / 252  # 1995-01 to 2015-12 = 252 months
        cv = np.std(all_values) / np.mean(all_values) if len(all_values) > 1 and np.mean(all_values) > 0 else np.nan
        zero_frac = zero_count / len(all_values) if len(all_values) > 0 else np.nan
        mean_q = np.mean(all_values) if all_values else np.nan

        # 分析缺失模式：连续缺失段
        gap_lengths = []
        current_gap = 0
        in_period = False
        for line in lines[data_start:]:
            line = line.strip().strip('"')
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                year = int(parts[0][:4])
            except:
                continue
            if 1995 <= year <= 2015:
                mean_val = parts[1].strip() if len(parts) > 1 else "NA"
                if mean_val in ("NA", "", "nan"):
                    current_gap += 1
                else:
                    if current_gap > 0:
                        gap_lengths.append(current_gap)
                    current_gap = 0
        if current_gap > 0:
            gap_lengths.append(current_gap)

        max_gap = max(gap_lengths) if gap_lengths else 0
        n_gaps = len(gap_lengths)

        records.append({
            "station_id": sid, "country": country,
            "lat": lat, "lon": lon, "area": area,
            "valid_months": valid_months_95_15,
            "completeness": completeness,
            "mean_q": mean_q, "cv": cv,
            "zero_frac": zero_frac,
            "max_gap": max_gap, "n_gaps": n_gaps,
            "n_values": len(all_values)
        })
    except Exception as e:
        pass

df = pd.DataFrame(records)
print(f"\n成功解析站点数: {len(df)}")

# ── 分类统计 ──
print(f"\n--- 完整度分布 ---")
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
labels = ["<10%", "10-30%", "30-50%", "50-70%", "70-90%", ">=90%"]
df["comp_bin"] = pd.cut(df["completeness"], bins=bins, labels=labels, right=False)
print(df["comp_bin"].value_counts().sort_index())

print(f"\n--- 按大洲/区域 ---")
continent_map = {
    "US": "N.America", "CA": "N.America", "MX": "N.America",
    "BR": "S.America", "AR": "S.America", "CO": "S.America", "VE": "S.America",
    "AU": "Oceania", "NZ": "Oceania",
    "FR": "Europe", "DE": "Europe", "ES": "Europe", "GB": "Europe", "IT": "Europe",
    "NO": "Europe", "AT": "Europe", "FI": "Europe", "CH": "Europe", "SE": "Europe",
    "IE": "Europe", "PL": "Europe", "NL": "Europe", "BE": "Europe", "DK": "Europe",
    "CZ": "Europe", "HU": "Europe", "SK": "Europe", "RO": "Europe", "PT": "Europe",
    "SI": "Europe", "IS": "Europe", "LT": "Europe", "LV": "Europe", "EE": "Europe",
    "BG": "Europe", "RS": "Europe", "GR": "Europe", "HR": "Europe", "UA": "Europe",
    "BY": "Europe", "RU": "Europe", "TR": "Europe",
    "JP": "Asia", "CN": "Asia", "IN": "Asia", "TH": "Asia", "MY": "Asia",
    "ID": "Asia", "KH": "Asia", "LA": "Asia", "VN": "Asia", "KR": "Asia",
    "PH": "Asia", "NP": "Asia", "KZ": "Asia", "KG": "Asia", "GE": "Asia",
    "AF": "Asia", "MM": "Asia",
    "ZA": "Africa", "ZW": "Africa", "TZ": "Africa", "MW": "Africa", "ML": "Africa",
    "SZ": "Africa", "NG": "Africa", "LS": "Africa", "BJ": "Africa", "GH": "Africa",
    "SN": "Africa", "NE": "Africa", "CF": "Africa", "GN": "Africa", "TD": "Africa",
    "BF": "Africa", "ET": "Africa", "MZ": "Africa", "ZM": "Africa", "NA": "Africa",
    "CI": "Africa", "BW": "Africa", "CD": "Africa",
}
df["continent"] = df["country"].map(continent_map).fillna("Other")
cont_stats = df.groupby("continent").agg(
    n_stations=("station_id", "count"),
    mean_completeness=("completeness", "mean"),
    median_completeness=("completeness", "median"),
    mean_area=("area", "median"),
    zero_frac_mean=("zero_frac", "mean"),
).round(3)
print(cont_stats)

print(f"\n--- 流量量级分布 ---")
q_bins = [0, 0.1, 1, 10, 100, 1000, 1e6]
q_labels = ["<0.1", "0.1-1", "1-10", "10-100", "100-1k", ">1k"]
df["q_bin"] = pd.cut(df["mean_q"], bins=q_bins, labels=q_labels, right=False)
print(df["q_bin"].value_counts().sort_index())

print(f"\n--- 零流量站点 ---")
print(f"零流量占比 > 50% 的站点数: {(df['zero_frac'] > 0.5).sum()} / {len(df)}")
print(f"零流量占比 > 30% 的站点数: {(df['zero_frac'] > 0.3).sum()} / {len(df)}")
print(f"零流量占比 > 10% 的站点数: {(df['zero_frac'] > 0.1).sum()} / {len(df)}")

print(f"\n--- 最大连续缺失段分布 ---")
gap_bins = [0, 1, 3, 6, 12, 24, 60, 300]
gap_labels = ["0(完整)", "1-3mo", "4-6mo", "7-12mo", "13-24mo", "25-60mo", ">60mo"]
df["gap_bin"] = pd.cut(df["max_gap"], bins=gap_bins, labels=gap_labels, right=True, include_lowest=True)
print(df["gap_bin"].value_counts().sort_index())

print(f"\n--- CV (变异系数) 分布 ---")
cv_valid = df["cv"].dropna()
print(f"  mean CV: {cv_valid.mean():.2f}")
print(f"  median CV: {cv_valid.median():.2f}")
print(f"  CV > 2 的站点比例: {(cv_valid > 2).sum() / len(cv_valid) * 100:.1f}%")
print(f"  CV > 3 的站点比例: {(cv_valid > 3).sum() / len(cv_valid) * 100:.1f}%")

# ── 2. GRDC 验证失败分析 ──
print("\n" + "=" * 70)
print("2. GRDC 外部验证失败分析")
print("=" * 70)

grdc = pd.read_csv(GRDC_CSV)
print(f"GRDC 验证站点数: {len(grdc)}")
print(f"  fill_NSE 有效: {grdc['fill_NSE'].notna().sum()}")

grdc_valid = grdc[grdc["fill_NSE"].notna()].copy()
print(f"\n--- fill_NSE 分布 ---")
print(f"  mean: {grdc_valid['fill_NSE'].mean():.3f}")
print(f"  median: {grdc_valid['fill_NSE'].median():.3f}")
print(f"  NSE > 0.5: {(grdc_valid['fill_NSE'] > 0.5).sum()} / {len(grdc_valid)} ({(grdc_valid['fill_NSE'] > 0.5).mean()*100:.1f}%)")
print(f"  NSE > 0: {(grdc_valid['fill_NSE'] > 0).sum()} / {len(grdc_valid)} ({(grdc_valid['fill_NSE'] > 0).mean()*100:.1f}%)")
print(f"  NSE < 0: {(grdc_valid['fill_NSE'] < 0).sum()} / {len(grdc_valid)} ({(grdc_valid['fill_NSE'] < 0).mean()*100:.1f}%)")
print(f"  NSE < -1: {(grdc_valid['fill_NSE'] < -1).sum()} / {len(grdc_valid)}")

print(f"\n--- 失败站点特征 (fill_NSE < 0) ---")
fail = grdc_valid[grdc_valid["fill_NSE"] < 0]
ok = grdc_valid[grdc_valid["fill_NSE"] >= 0]
print(f"  失败站 mean_grdc 中位数: {fail['mean_grdc'].median():.2f}")
print(f"  成功站 mean_grdc 中位数: {ok['mean_grdc'].median():.2f}")
print(f"  失败站 n_filled 中位数: {fail['n_filled'].median():.0f}")
print(f"  成功站 n_filled 中位数: {ok['n_filled'].median():.0f}")
print(f"  失败站 max_gap 中位数: {fail['max_gap_months'].median():.0f}")
print(f"  成功站 max_gap 中位数: {ok['max_gap_months'].median():.0f}")

# PBias 分析
print(f"\n--- PBias (百分比偏差) 分析 ---")
print(f"  |PBias| > 50%: {(grdc_valid['fill_PBias'].abs() > 50).sum()} / {len(grdc_valid)}")
print(f"  |PBias| > 100%: {(grdc_valid['fill_PBias'].abs() > 100).sum()} / {len(grdc_valid)}")
print(f"  PBias > 0 (高估): {(grdc_valid['fill_PBias'] > 0).sum()}")
print(f"  PBias < 0 (低估): {(grdc_valid['fill_PBias'] < 0).sum()}")

# 量级偏差
print(f"\n--- 量级偏差 (mean_gsim vs mean_grdc) ---")
grdc["ratio"] = grdc["mean_gsim"] / grdc["mean_grdc"]
print(f"  ratio 中位数: {grdc['ratio'].median():.2f}")
print(f"  ratio > 2 (GSIM 高估 2x+): {(grdc['ratio'] > 2).sum()}")
print(f"  ratio < 0.5 (GSIM 低估 2x+): {(grdc['ratio'] < 0.5).sum()}")

# 小流量站问题
print(f"\n--- 小流量站 (mean_grdc < 1 m3/s) ---")
small = grdc_valid[grdc_valid["mean_grdc"] < 1]
print(f"  数量: {len(small)}")
if len(small) > 0:
    print(f"  fill_NSE 中位数: {small['fill_NSE'].median():.3f}")
    print(f"  |PBias| 中位数: {small['fill_PBias'].abs().median():.1f}%")

print(f"\n--- 极端失败案例 (fill_NSE < -10) ---")
extreme = grdc_valid[grdc_valid["fill_NSE"] < -10].sort_values("fill_NSE")
for _, row in extreme.head(10).iterrows():
    print(f"  {row['gsim_id']} ({row['name']}): NSE={row['fill_NSE']:.1f}, "
          f"mean_grdc={row['mean_grdc']:.2f}, mean_gsim={row['mean_gsim']:.2f}, "
          f"n_filled={row['n_filled']}, max_gap={row['max_gap_months']}")

# ── 3. 核心问题诊断 ──
print("\n" + "=" * 70)
print("3. 核心问题诊断")
print("=" * 70)

# 问题1: GSIM 原始数据本身就和 GRDC 不一致
print("\n[问题1] GSIM vs GRDC 原始数据一致性")
print(f"  all_NSE (全序列含原始+插补): mean={grdc['all_NSE'].mean():.3f}, median={grdc['all_NSE'].median():.3f}")
print(f"  fill_NSE (仅插补部分): mean={grdc_valid['fill_NSE'].mean():.3f}, median={grdc_valid['fill_NSE'].median():.3f}")
print(f"  → all_NSE 很高说明 GSIM 原始观测和 GRDC 基本一致")
print(f"  → fill_NSE 很低说明问题出在插补方法上")

# 问题2: 插补月数 vs 性能
print(f"\n[问题2] 插补月数 vs 性能")
for n_range, label in [(range(1, 3), "1-2月"), (range(3, 7), "3-6月"), (range(7, 15), "7-14月"), (range(15, 300), "15+月")]:
    subset = grdc_valid[grdc_valid["n_filled"].isin(n_range)]
    if len(subset) > 0:
        print(f"  {label}: n={len(subset)}, median NSE={subset['fill_NSE'].median():.3f}")

print("\n完成!")
