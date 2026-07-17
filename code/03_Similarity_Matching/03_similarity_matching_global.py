import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from gsim_plus_config import K_NEIGHBORS, STEP1_DIR, STEP2_DIR, STEP3_DIR


GROUP_WEIGHTS = {
    "hydro": 0.40,
    "climate": 0.25,
    "topo": 0.15,
    "spatial": 0.10,
    "soil": 0.10,
}

FEATURE_PRIORS = {
    "hydro": {
        "mean_flow_m3s": 0.50,
        "upstream_area_km2": 0.35,
        "area_local_hybas_km2": 0.15,
    },
    "climate": {
        "tp_mean": 0.24,
        "tp_std": 0.20,
        "tp_cv": 0.16,
        "t2m_mean": 0.16,
        "t2m_std": 0.12,
        "e_mean": 0.08,
        "e_std": 0.04,
    },
    "topo": {
        "altitude": 0.45,
        "slope": 0.55,
    },
    "spatial": {
        "latitude": 0.50,
        "longitude": 0.50,
    },
    "soil": {
        "snd": 0.34,
        "slt": 0.33,
        "scl": 0.33,
    },
}

HYDRO_LOG_COLS = {"mean_flow_m3s", "upstream_area_km2", "area_local_hybas_km2"}


def cdist(XA, XB):
    diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def load_inputs():
    feature_file = STEP2_DIR / "station_features_with_meteo.csv"
    if not feature_file.exists():
        feature_file = STEP2_DIR / "station_features.csv"
    features = pd.read_csv(feature_file)
    anchor_df = pd.read_csv(STEP1_DIR / "anchor_stations.csv")
    target_df = pd.read_csv(STEP1_DIR / "target_stations.csv")

    anchor_ids = set(anchor_df["station_id"])
    target_ids = set(target_df["station_id"])

    anchor_features = features[features["station_id"].isin(anchor_ids)].copy()
    target_features = features[features["station_id"].isin(target_ids)].copy()
    return features, anchor_features, target_features, target_ids


def get_feature_groups(features):
    groups = {
        "climate": ["tp_mean", "tp_std", "tp_cv", "t2m_mean", "t2m_std", "e_mean", "e_std"],
        "topo": ["altitude", "slope"],
        "soil": ["snd", "slt", "scl"],
        "spatial": ["latitude", "longitude"],
        "hydro": ["mean_flow_m3s", "area_local_hybas_km2", "upstream_area_km2"],
    }
    return {k: [f for f in v if f in features.columns] for k, v in groups.items()}


def learn_weights(features, feature_groups):
    final_weights = {}
    for group_name, feats in feature_groups.items():
        if not feats:
            continue
        g_weight = GROUP_WEIGHTS.get(group_name, 0.0)
        priors = FEATURE_PRIORS.get(group_name, {})
        feat_weights = {feat: float(priors.get(feat, 1.0)) for feat in feats}
        total = sum(feat_weights.values())
        for feat in feats:
            final_weights[feat] = g_weight * feat_weights[feat] / total

    total = sum(final_weights.values())
    return {k: v / total for k, v in final_weights.items()}


def calculate_similarity(anchor_features, target_features, weights):
    cols = list(weights.keys())
    anchor_numeric = anchor_features[cols].copy()
    target_numeric = target_features[cols].copy()
    for col in HYDRO_LOG_COLS:
        if col in anchor_numeric.columns:
            anchor_numeric[col] = np.log1p(pd.to_numeric(anchor_numeric[col], errors="coerce").clip(lower=0))
            target_numeric[col] = np.log1p(pd.to_numeric(target_numeric[col], errors="coerce").clip(lower=0))
    scaler = StandardScaler()
    anchor_matrix = anchor_numeric.fillna(anchor_numeric.median(numeric_only=True)).fillna(0).values
    target_matrix = target_numeric.fillna(anchor_numeric.median(numeric_only=True)).fillna(0).values
    anchor_scaled = scaler.fit_transform(anchor_matrix)
    target_scaled = scaler.transform(target_matrix)
    weight_vec = np.array([weights[c] for c in cols])
    distances = cdist(target_scaled * weight_vec, anchor_scaled * weight_vec)
    return distances


def build_topk(anchor_features, target_features, distances):
    anchor_ids = anchor_features["station_id"].values
    target_ids = target_features["station_id"].values
    anchor_kg_major = anchor_features["kg_major"].fillna("").astype(str).values if "kg_major" in anchor_features.columns else np.array([""] * len(anchor_features))
    anchor_kg_code = anchor_features["kg_code"].fillna("").astype(str).values if "kg_code" in anchor_features.columns else np.array([""] * len(anchor_features))
    anchor_continent = anchor_features["hybas_source"].fillna("").astype(str).values if "hybas_source" in anchor_features.columns else np.array([""] * len(anchor_features))
    rows = []
    for i, target_id in enumerate(target_ids):
        target_major = ""
        target_code = ""
        target_continent = ""
        if "kg_major" in target_features.columns:
            target_major = str(target_features.iloc[i].get("kg_major", "") or "")
        if "kg_code" in target_features.columns:
            target_code = str(target_features.iloc[i].get("kg_code", "") or "")
        if "hybas_source" in target_features.columns:
            target_continent = str(target_features.iloc[i].get("hybas_source", "") or "")

        same_continent = np.where(anchor_continent == target_continent)[0] if target_continent else np.array([], dtype=int)
        same_major = np.where(anchor_kg_major == target_major)[0] if target_major else np.array([], dtype=int)
        same_code_prefix = np.where(np.array([a[:1] for a in anchor_kg_code]) == target_code[:1])[0] if target_code else np.array([], dtype=int)

        same_continent_major = np.intersect1d(same_continent, same_major) if len(same_continent) and len(same_major) else np.array([], dtype=int)
        same_continent_code = np.intersect1d(same_continent, same_code_prefix) if len(same_continent) and len(same_code_prefix) else np.array([], dtype=int)

        if len(same_continent_major) >= K_NEIGHBORS:
            candidate_idx = same_continent_major
            climate_filter = "continent+kg_major"
        elif len(same_continent_code) >= K_NEIGHBORS:
            candidate_idx = same_continent_code
            climate_filter = "continent+kg_code_prefix"
        elif len(same_continent) > 0:
            candidate_idx = same_continent
            climate_filter = "continent_only"
        elif len(same_major) >= K_NEIGHBORS:
            candidate_idx = same_major
            climate_filter = "kg_major"
        elif len(same_code_prefix) >= K_NEIGHBORS:
            candidate_idx = same_code_prefix
            climate_filter = "kg_code_prefix"
        else:
            candidate_idx = np.arange(len(anchor_ids))
            climate_filter = "fallback_all"

        local_order = candidate_idx[np.argsort(distances[i, candidate_idx])[:K_NEIGHBORS]]
        for rank, j in enumerate(local_order, start=1):
            distance = float(distances[i, j])
            similarity = 1 / (1 + distance)
            rows.append(
                {
                    "target_station": target_id,
                    "anchor_station": anchor_ids[j],
                    "rank": rank,
                    "distance": round(distance, 4),
                    "similarity": round(float(similarity), 4),
                    "climate_filter": climate_filter,
                }
            )
    return pd.DataFrame(rows)


def main():
    features, anchor_features, target_features, _ = load_inputs()
    feature_groups = get_feature_groups(features)
    weights = learn_weights(features, feature_groups)
    distances = calculate_similarity(anchor_features, target_features, weights)
    topk = build_topk(anchor_features, target_features, distances)

    topk.to_csv(STEP3_DIR / f"top_{K_NEIGHBORS}_similar_stations.csv", index=False)
    pd.DataFrame(list(weights.items()), columns=["feature", "weight"]).sort_values(
        "weight", ascending=False
    ).to_csv(STEP3_DIR / "feature_weights.csv", index=False)

    summary = {
        "n_anchor": int(len(anchor_features)),
        "n_target": int(len(target_features)),
        "k_neighbors": K_NEIGHBORS,
        "mean_similarity": float(topk["similarity"].mean()),
        "std_similarity": float(topk["similarity"].std()),
    }
    with open(STEP3_DIR / "similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved to: {STEP3_DIR}")


if __name__ == "__main__":
    main()
