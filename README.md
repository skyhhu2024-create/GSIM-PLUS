# GSIM-PLUS: A Global Monthly Streamflow Gap-Filling Dataset (1995–2015)

**SGML (Similarity-Guided Meta-Learning)** — An adaptive basin-similarity-driven meta-learning framework for global streamflow gap-filling.

> Target journal: *Earth System Science Data* (ESSD)

---

## Overview

GSIM-PLUS is a comprehensive pipeline for gap-filling monthly streamflow records from the **Global Streamflow Indices and Metadata (GSIM)** archive. It produces a globally consistent, quality-flagged streamflow dataset spanning **1995–2015 (252 months)** by combining physically meaningful catchment similarity matching with MAML-based meta-learning.

### Key Numbers

| Item | Value |
|------|-------|
| Total GSIM stations analysed | 30,960 |
| Anchor stations (≥ 90 % coverage) | 7,323 |
| Target stations (30–90 % coverage) | 8,731 |
| Study period | 1995-01 to 2015-12 (252 months) |
| Gap-filled data points | 277,729 |
| Similarity neighbours (K) | 5 |
| Features used | 17 (hydro + climate + topo + spatial + soil) |

---

## Method

The core method (**SGML**) consists of two stages:

1. **Similarity-Guided Donor Selection** — For each target station, the top-K most similar anchor stations are identified using a weighted multi-group feature distance (hydrological, climate, topographic, spatial, and soil attributes).

2. **Meta-Learning Gap-Filling** — A MAML (Model-Agnostic Meta-Learning) model, enhanced with a DonorTrend correction, is trained on anchor station data and rapidly adapted to each target station via few-shot fine-tuning.

### Compared Methods (8 total)

| Method | Type |
|--------|------|
| **MAML DonorTrend** | Meta-learning + trend correction (proposed) |
| MAML | Meta-learning baseline |
| Random Forest | Machine learning |
| Linear | Statistical |
| LSTM | Deep learning |
| KNN | Instance-based |
| Seasonal Mean | Climatological |
| IDW | Spatial interpolation |

---

## Pipeline

The pipeline consists of 8 sequential steps, orchestrated by `run_all_steps.py`:

```
GSIM raw monthly data (30,960 stations)
  │
  ├─ Step 1: Station Selection
  │    └─ Classify into anchor / target / insufficient
  │
  ├─ Step 2: Feature Extraction
  │    └─ 17 features: hydro (3) + climate (7) + topo (2) + spatial (2) + soil (3)
  │
  ├─ Step 3: Similarity Matching
  │    └─ Weighted distance → top-5 donor stations per target
  │
  ├─ Step 4: Random 30% Validation
  ├─ Step 5: Continuous Gap Validation (3 / 6 / 12 months)
  ├─ Step 6: Hybrid Scenario Validation (H1 / H2 / H3)
  ├─ Step 7: Super-Long Gap Analysis (25+ months)
  │
  └─ Step 8: Product Generation
       └─ GSIM-PLUS dataset with quality flags (Q0–Q3)
```

### Directory Structure

```
GSIM-PLUS/
├── 00_common/                          # Shared utilities & config
│   ├── gsim_plus_config.py             # Central configuration
│   ├── gsim_core.py                    # Core ML functions (MAML, LSTM, RF, etc.)
│   ├── gsim_plus_utils.py              # Data I/O utilities
│   ├── validation_wrappers.py          # Validation framework
│   └── model_cache/                    # Pre-trained model cache
│
├── 01_Station_Selection/               # Step 1
│   ├── 01_station_selector_global.py
│   ├── anchor_stations.csv             # 7,323 anchor stations
│   ├── target_stations.csv             # 8,731 target stations
│   └── all_stations_1995_2015.csv      # Full station inventory
│
├── 02_Feature_Table/                   # Step 2
│   ├── 02_build_feature_table_global.py
│   ├── 02_extract_meteo_features_global.py
│   ├── 02_extract_hydro_features_global.py
│   └── station_features_with_meteo.csv # Final 17-feature table
│
├── 03_Similarity_Matching/             # Step 3
│   ├── 03_similarity_matching_global.py
│   ├── feature_weights.csv             # Learned feature importance
│   └── top_5_similar_stations.csv      # Donor-target pairs
│
├── 04_Random_30pct_Validation/         # Step 4
│   ├── 04_random_30pct_validation_noleak.py
│   ├── random_30pct_summary.csv
│   └── random_30pct/                   # Per-method prediction CSVs
│
├── 05_Continuous_Gap_Validation/       # Step 5
│   ├── 05_continuous_gap_validation_noleak.py
│   ├── continuous_gap_summary.csv
│   └── 3_months/ 6_months/ 12_months/
│
├── 06_Hybrid_Validation/               # Step 6
│   ├── 06_hybrid_gap_validation_h123_noleak.py
│   ├── hybrid_h123_summary.csv
│   ├── hybrid_h123_summary_excluding_abnormal.csv
│   └── H1_sparse_dominant/ H2_balanced_mixed/ H3_long_gap_dominant/
│
├── 07_SuperLong_Gap_Analysis/          # Step 7
│   ├── 07_super_long_gap_analysis.py
│   ├── super_long_summary_excluding_abnormal.csv
│   └── super_long_25plus/
│
├── 08_GSIM_PLUS_Product/              # Step 8: Final product
│   ├── 08_build_gsim_plus_dataset.py
│   ├── 08_build_gsim_plus_anchor_dataset.py
│   ├── maml_donor_trend_guarded/       # Target station products
│   │   └── GSIM_fill/                  # 8,731 individual station CSVs
│   └── MAML_DonorTrend_Guarded/        # Anchor station products
│       └── GSIM_fill_anchor/           # 7,323 individual station CSVs
│
├── 111-paper/                          # Paper figures
│   ├── Fig1_global_station_overview.py
│   ├── Fig3_feature_similarity.py
│   ├── Fig3_random_validation.py
│   ├── Fig4_continuous_degradation.py
│   ├── Fig5_hybrid_heatmap.py
│   ├── Fig6_superlong_analysis.py
│   ├── Fig8_koppen_boxplot.py
│   ├── Fig9_product_quality.py
│   └── Fig10_timeseries.py
│
├── 999 material/                       # Input data (not tracked)
│   ├── GSIM_attribute.csv
│   ├── Global_1995-2015.nc
│   └── Attribute/                      # Shapefiles (HYBAS, HydroRIVERS)
│
├── run_all_steps.py                    # Pipeline orchestrator
└── README.md
```

---

## Features

### 17 Catchment Similarity Features (5 Groups)

| Group | Weight | Features |
|-------|--------|----------|
| **Hydrological** | 40 % | Mean flow, Upstream area, Local basin area |
| **Climate** | 25 % | Precipitation (mean, std, CV), Temperature (mean, std), Evaporation (mean, std) |
| **Topography** | 15 % | Altitude, Slope |
| **Spatial** | 10 % | Latitude, Longitude |
| **Soil** | 10 % | Sand, Silt, Clay content |

### Quality Flags

Each gap-filled data point is assigned a quality flag based on the gap length:

| Flag | Gap Length | Description |
|------|-----------|-------------|
| Q0 | 0 | Original observed value |
| Q1 | 1–3 months | Short gap, high confidence |
| Q2 | 4–24 months | Medium gap |
| Q3 | > 24 months | Long gap, lower confidence |

---

## Validation Framework

Four complementary validation experiments ensure robustness:

### 1. Random 30 % Withholding (Step 4)

30 % of observed data points are randomly masked and predicted. No data leakage between train/test.

### 2. Continuous Gap Degradation (Step 5)

Consecutive gaps of 3, 6, and 12 months are artificially created to evaluate performance degradation with increasing gap length.

### 3. Hybrid Scenario Validation (Step 6)

Three realistic gap-pattern scenarios based on different gap-length distributions:

| Scenario | Description | Gap distribution |
|----------|-------------|-----------------|
| H1 | Sparse dominant | Mostly short gaps (1–3 months) |
| H2 | Balanced mixed | Even mix of gap lengths |
| H3 | Long gap dominant | Mostly long gaps (13–25+ months) |

### 4. Super-Long Gap Analysis (Step 7)

Focused evaluation on gaps exceeding 25 months — the most challenging scenario for any gap-filling method.

---

## Results Summary

### Random 30 % Validation (Step 4)

| Method | NSE | KGE |
|--------|-----|-----|
| **MAML DonorTrend** | **0.865** | **0.920** |
| Random Forest | 0.763 | 0.829 |
| MAML | 0.761 | 0.801 |
| Linear | 0.747 | 0.809 |
| LSTM | 0.631 | 0.724 |
| KNN | 0.625 | 0.802 |
| Seasonal Mean | 0.587 | 0.742 |
| IDW | 0.585 | 0.783 |

### Continuous Gap Validation — NSE (Step 5)

| Method | 3 months | 6 months | 12 months |
|--------|----------|----------|-----------|
| **MAML DonorTrend** | **0.879** | **0.929** | **0.795** |
| MAML | 0.740 | 0.713 | 0.570 |
| Linear | 0.720 | 0.671 | 0.484 |
| Random Forest | 0.718 | 0.577 | 0.453 |

### Super-Long Gap (25+ months) Validation (Step 7)

| Method | NSE | KGE |
|--------|-----|-----|
| **MAML DonorTrend** | **0.511** | **0.719** |
| MAML | 0.281 | 0.598 |
| LSTM | 0.251 | 0.595 |
| Linear | 0.228 | 0.586 |

---

## Product Output Format

Each station CSV file (`{STATION_ID}.csv`) contains:

| Column | Description |
|--------|-------------|
| `station_id` | GSIM station identifier |
| `date` | Monthly timestamp (YYYY-MM-DD) |
| `year` | Year |
| `month` | Month |
| `observed_streamflow` | Original GSIM value (blank if missing) |
| `final_streamflow` | Gap-filled or observed value (m³/s) |
| `segment_length` | Length of the gap segment (0 = observed) |
| `fill_method` | `OBSERVED` or method name |
| `quality_flag` | Q0 / Q1 / Q2 / Q3 |

---

## Requirements

### Python Environment

- Python 3.8+
- PyTorch (with CUDA support recommended)
- scikit-learn
- pandas, numpy
- matplotlib
- cartopy (for global maps)
- geopandas (for spatial feature extraction)

### Input Data

- GSIM monthly streamflow time series (`.mon` files)
- `GSIM_attribute.csv` — station metadata
- `Global_1995-2015.nc` — ERA5 reanalysis (precipitation, temperature, evaporation)
- HydroRIVERS v10 shapefiles
- HydroBASINS (HYBAS) level-05 shapefiles
- `stations_full_static_with_climate_kg_feat.csv` — static features with Koppen-Geiger classification

---

## Usage

### Run the Full Pipeline

```bash
python run_all_steps.py
```

This sequentially executes Steps 1–8. Each step reads outputs from previous steps and writes to its own directory.

### Run Individual Steps

```bash
# Step 1: Station classification
python 01_Station_Selection/01_station_selector_global.py

# Step 2: Feature extraction
python 02_Feature_Table/02_build_feature_table_global.py
python 02_Feature_Table/02_extract_meteo_features_global.py
python 02_Feature_Table/02_extract_hydro_features_global.py

# Step 3: Similarity matching
python 03_Similarity_Matching/03_similarity_matching_global.py

# Steps 4-7: Validation experiments
python 04_Random_30pct_Validation/04_random_30pct_validation_noleak.py
python 05_Continuous_Gap_Validation/05_continuous_gap_validation_noleak.py
python 06_Hybrid_Validation/06_hybrid_gap_validation_h123_noleak.py
python 07_SuperLong_Gap_Analysis/07_super_long_gap_analysis.py

# Step 8: Generate GSIM-PLUS product
python 08_GSIM_PLUS_Product/08_build_gsim_plus_dataset.py
```

### Generate Paper Figures

```bash
cd 111-paper
python Fig1_global_station_overview.py
python Fig3_feature_similarity.py
python Fig3_random_validation.py
python Fig4_continuous_degradation.py
python Fig5_hybrid_heatmap.py
python Fig6_superlong_analysis.py
python Fig8_koppen_boxplot.py
python Fig9_product_quality.py
python Fig10_timeseries.py
```

---

## Configuration

All pipeline parameters are centrally defined in `00_common/gsim_plus_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STUDY_START_YEAR` | 1995 | Start of study period |
| `STUDY_END_YEAR` | 2015 | End of study period |
| `ANCHOR_THRESHOLD` | 0.90 | Minimum data coverage for anchor stations |
| `TARGET_MIN` | 0.30 | Minimum data coverage for target stations |
| `TARGET_MAX` | 0.90 | Maximum data coverage for target stations |
| `EVALUABLE_MIN_MONTHS` | 120 | Minimum observed months for validation |
| `K_NEIGHBORS` | 5 | Number of similar donor stations |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |
| `SUPER_LONG_MIN_MONTHS` | 25 | Threshold for super-long gaps |

---

## Citation

If you use GSIM-PLUS in your research, please cite:

```
[Citation to be added upon publication]
```

---

## License

[License to be determined]

---

## Acknowledgements

- **GSIM**: Do, H. X. et al. (2018). The Global Streamflow Indices and Metadata Archive (GSIM). *Water Resources Research*.
- **ERA5**: Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*.
- **HydroRIVERS / HydroBASINS**: Lehner, B., Grill, G. (2013). *Hydrological Processes*.
