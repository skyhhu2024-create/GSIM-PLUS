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

---

## Key Numbers

| Item | Value |
|---|---|
| Total GSIM stations analysed | 30,959 |
| Anchor stations | 7,323 |
| Target stations | 8,731 |
| Study period | 1995-01 to 2015-12 (252 months) |
| Target infilled monthly records | 277,729 |
| Anchor infilled monthly records | 25,542 |
| Similarity neighbours (K) | 5 |
| Catchment similarity features | 17 |
| Low-flow safeguard | median flow < 0.02 m3 s-1 |
| Target safeguard output | 676 stations; 33,542 MAML-filled months |
| Anchor safeguard output | 212 stations; 1,823 MAML-filled months |
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

The primary production method is **DTRR**. In the guarded production workflow, stations with median flow below 0.02 m3 s-1 are automatically routed to `MAML` to avoid the numerical divergence observed for DTRR in very-low-flow stress tests. This cutoff is a pragmatic production safeguard rather than a theoretically unique threshold. As a result:

- `DTRR` is the default and dominant fill method.
- `MAML` appears only as a fallback for low-flow stations.

Both production paths enforce the physical boundary `streamflow >= 0` before a reconstructed value is written back as recursive lag-1 input.

This constraint applies only to reconstructed values. Rows labelled `Q0` preserve the source GSIM observations unchanged, including any source-data anomalies.

### MAML Low-Flow Safeguard

**Model-Agnostic Meta-Learning (MAML)** is implemented as a neural regression model with seasonal terms and lag-1 target flow as inputs. Its meta-initialization is learned exclusively from anchor stations. During meta-training, random 30 % gaps and continuous gaps of 3, 6, 12, and 25-48 months are simulated at anchor stations. The remaining observations form the support set, while the hidden months form a recursively predicted query set. Station-specific standardization is calculated from support observations only.

For application to a target station, the learned initialization is adapted using observations from its matched anchor stations and its own available record. Missing months are then reconstructed sequentially, with each prediction available as the lag-1 input for the next missing month. All reconstructed flows are constrained to be nonnegative before recursive feedback. MAML is therefore also recursive; its use as the safeguard is supported by its empirical numerical stability and by its consistently highest NSE among the non-DTRR methods across the evaluated validation scenarios.

---

## Validation Summary

Validation results show that DTRR provides the highest general performance, while MAML has the highest NSE among the non-DTRR methods in every evaluated scenario and remains numerically stable in the very-low-flow settings where guarded fallback is required.

### Random 30 % Validation

Source: `results/final_validation_summary.csv` (`scenario = random_30pct`)

| Method | NSE | KGE |
|---|---:|---:|
| **DTRR** | **0.8642** | **0.9197** |
| MAML | 0.8080 | 0.8611 |
| Random Forest | 0.7631 | 0.8287 |
| Linear | 0.7467 | 0.8092 |

### Continuous Gap Validation

Source: `results/final_validation_summary.csv` (`scenario = 3_months, 6_months, 12_months`)

| Gap length | DTRR NSE | DTRR KGE | MAML NSE | MAML KGE |
|---|---:|---:|---:|---:|
| 3 months | 0.8794 | 0.8652 | 0.7854 | 0.8117 |
| 6 months | 0.9295 | 0.9308 | 0.7469 | 0.8545 |
| 12 months | 0.7975 | 0.8690 | 0.6257 | 0.7314 |

### Hybrid Validation

Source: `results/final_validation_summary.csv` (the excluded-station rows for H2 and H3)

| Scenario | DTRR NSE | DTRR KGE | MAML NSE | MAML KGE |
|---|---:|---:|---:|---:|
| H1 short-gap dominant | 0.7738 | 0.8614 | 0.7246 | 0.7932 |
| H2 balanced mixed | 0.8090 | 0.8806 | 0.6976 | 0.7798 |
| H3 long-gap dominant | 0.7290 | 0.8239 | 0.6463 | 0.7527 |

### Super-Long Gap Validation

Source: `results/final_validation_summary.csv` (the row excluding `AU_0000492`, `AU_0002125`, and `BR_0000375`)

| Method | NSE | KGE |
|---|---:|---:|
| **DTRR** | **0.5109** | **0.7153** |
| MAML | 0.2862 | 0.6073 |
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
- `09 GRDC.../` for comparison against independent GRDC observations
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
- GRDC independent-reference comparison figure scripts

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
| `kg_major`, `kg_code` | Koppen-Geiger climate context |
| `arid_flag` | Whether the station is in the major arid climate class |
| `low_flow_flag` | Whether station median flow is below 0.02 m3 s-1 |
| `context_risk_flag` | Combined `ARID` and/or `LOW_FLOW` context tag |
| `guard_applied` | Whether the MAML low-flow safeguard was activated |

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

The raw GSIM records and large external geospatial and climate inputs are not included. Configure their locations before running the workflow. The main supported environment variables are:

- `GSIM_PLUS_PROJECT_DIR`: working directory for intermediate and final outputs.
- `GSIM_MONTHLY_DIR`: directory containing the GSIM monthly station files.
- `GSIM_PLUS_MATERIAL_DIR`: directory containing attributes, climate inputs, and geospatial material.

For example, in PowerShell:

```powershell
$env:GSIM_PLUS_PROJECT_DIR = "D:\\path\\to\\gsim-plus-workspace"
$env:GSIM_MONTHLY_DIR = "D:\\path\\to\\GSIM_indices\\TIMESERIES\\monthly"
$env:GSIM_PLUS_MATERIAL_DIR = "D:\\path\\to\\gsim-plus-materials"
```

Run the full pipeline:

```bash
python code/run_all_steps.py
```

Run the final production scripts directly:

```bash
python code/08_GSIM_PLUS_Product/08_build_gsim_plus_dataset.py
python code/08_GSIM_PLUS_Product/08_build_gsim_plus_anchor_dataset.py
```

Run the MAML regression tests:

```bash
python code/tests/test_maml_training.py
```

The guarded DTRR workflow, the 0.02 m3 s-1 safeguard, and the contextual risk fields are the default configuration in both production scripts.

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
