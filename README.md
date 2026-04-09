# GSIM-PLUS: A Global Monthly Streamflow Gap-Filling Dataset (1995-2015)

GSIM-PLUS is a global monthly streamflow gap-filling workflow and data product built from the Global Streamflow Indices and Metadata (GSIM) archive. The current release uses **DTRR** as the primary reconstruction method, combined with similarity-guided donor selection and a guarded fallback to `MAML` for a small subset of very low-flow stations.

This repository is the code archive prepared for GitHub release. It includes the full computational workflow under `code/` and the manuscript figure scripts under `fig_code/`.

> Target journal: *Earth System Science Data* (ESSD)

---

## Overview

The GSIM-PLUS workflow consists of three connected components:

1. Station screening and feature construction for global GSIM stations.
2. Similarity-based donor identification using weighted hydro-climatic, topographic, spatial, and soil descriptors.
3. Monthly gap filling using **DTRR**, with `MAML` retained only as a low-flow fallback in the guarded production scheme.

The public method name is now **DTRR**. Earlier internal labels have been retired and are no longer used as release terminology in this code archive.

---

## Key Numbers

| Item | Value |
|---|---|
| Total GSIM stations analysed | 30,960 |
| Anchor stations | 7,323 |
| Target stations | 8,731 |
| Study period | 1995-01 to 2015-12 (252 months) |
| Target infilled monthly records | 277,729 |
| Anchor infilled monthly records | 25,542 |
| Similarity neighbours (K) | 5 |
| Catchment similarity features | 17 |
| Final production label | `DTRR + low_flow_guard` |

---

## Method

### Similarity-Guided Donor Selection

For each target station, the workflow identifies the Top-5 most similar anchor stations using a weighted multi-group distance built from 17 catchment attributes:

- hydrological
- climate
- topography
- spatial location
- soil texture

The default group weights are:

| Group | Weight |
|---|---|
| Hydrological | 40 % |
| Climate | 25 % |
| Topography | 15 % |
| Spatial | 10 % |
| Soil | 10 % |

### DTRR Gap Filling

The primary production method is **DTRR**. In the guarded production workflow, stations with very low median flow are automatically routed to `MAML` to avoid instability in recursive prediction. As a result:

- `DTRR` is the default and dominant fill method.
- `MAML` appears only as a fallback for low-flow stations.

---

## Validation Summary

The latest workflow outputs show that DTRR is consistently the best-performing method among the tested baselines across random masking, continuous gaps, and super-long-gap reconstruction.

### Random 30 % Validation

Source: `04_Random_30pct_Validation/random_30pct_summary.csv`

| Method | NSE | KGE |
|---|---:|---:|
| **DTRR** | **0.8646** | **0.9195** |
| Random Forest | 0.7631 | 0.8287 |
| MAML | 0.7606 | 0.8007 |
| Linear | 0.7467 | 0.8092 |

### Continuous Gap Validation

Source: `05_Continuous_Gap_Validation/continuous_gap_summary.csv`

| Gap length | DTRR NSE | DTRR KGE |
|---|---:|---:|
| 3 months | 0.8788 | 0.8646 |
| 6 months | 0.9292 | 0.9318 |
| 12 months | 0.7954 | 0.8689 |

### Hybrid Validation

Source: `06_Hybrid_Validation/hybrid_h123_summary_excluding_abnormal.csv`

| Scenario | DTRR NSE | DTRR KGE |
|---|---:|---:|
| H2 balanced mixed | 0.8089 | 0.8827 |
| H3 long-gap dominant | 0.6910 | 0.8189 |

### Super-Long Gap Validation

Source: `07_SuperLong_Gap_Analysis/super_long_summary_excluding_abnormal.csv`

| Method | NSE | KGE |
|---|---:|---:|
| **DTRR** | **0.5109** | **0.7185** |
| MAML | 0.2808 | 0.5976 |
| Linear | 0.2284 | 0.5855 |
| LSTM | 0.2510 | 0.5946 |

---

## Repository Structure

### `code/`

This directory contains the full computational workflow used for GSIM-PLUS generation and validation:

- `00_common/`
- `01_Station_Selection/`
- `02_Feature_Table/`
- `03_Similarity_Matching/`
- `04_Random_30pct_Validation/`
- `05_Continuous_Gap_Validation/`
- `06_Hybrid_Validation/`
- `07_SuperLong_Gap_Analysis/`
- `08_GSIM_PLUS_Product/`
- `09 GRDC.../` for GRDC-oriented external validation
- `run_all_steps.py`
- `analyze_gsim_distribution.py`

### `fig_code/`

This directory contains the plotting and manuscript-side statistics scripts used for the figures and figure-support analyses, including:

- global station overview
- similarity and random-validation figures
- continuous-gap and hybrid-validation figures
- super-long-gap analysis figures
- product-quality and timeseries figures
- Koppen climate-zone summary scripts
- GRDC independent-validation figure scripts

---

## Product Output

The final GSIM-PLUS product is generated under the guarded DTRR workflow:

- target product directory: `dtrr_guarded`
- anchor companion directory: `DTRR_Guarded_Anchor`

Each station CSV contains:

| Column | Description |
|---|---|
| `station_id` | GSIM station identifier |
| `date` | Monthly timestamp |
| `year` | Year |
| `month` | Month |
| `observed_streamflow` | Original GSIM value |
| `final_streamflow` | Final observed-or-filled value |
| `segment_length` | Length of the gap segment |
| `fill_method` | `OBSERVED`, `DTRR`, or `MAML` |
| `quality_flag` | `Q0`, `Q1`, `Q2`, `Q3`, or `Q4` |

Quality flags represent:

| Flag | Meaning |
|---|---|
| `Q0` | original observation |
| `Q1` | short gap fill (1-3 months) |
| `Q2` | medium gap fill (4-24 months) |
| `Q3` | long gap fill (25+ months) |
| `Q4` | unfilled or failed reconstruction |

---

## Running the Workflow

Run the full pipeline:

```bash
python run_all_steps.py
```

Run the final production scripts directly:

```bash
python code/08_GSIM_PLUS_Product/08_build_gsim_plus_dataset.py
python code/08_GSIM_PLUS_Product/08_build_gsim_plus_anchor_dataset.py
```

The guarded DTRR workflow is now the default configuration in both production scripts.

---

## Requirements

Main dependencies include:

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- geopandas
- cartopy

See `requirements.txt` for the packaged dependency list.

---

## Notes

- This code archive does not include the large raw GSIM inputs, shapefiles, NetCDF climate products, or the final released data tables.
- The README numbers above were updated from the latest summary files in the current workspace.
- `README_PIPELINE.md` provides a step-by-step pipeline summary focused on code execution rather than headline results.
