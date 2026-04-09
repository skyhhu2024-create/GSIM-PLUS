# GSIM-PLUS Pipeline

This document summarises the full GSIM-PLUS computational pipeline included under `code/`. The workflow builds the final monthly streamflow product for 1995-2015 and evaluates the DTRR method under multiple validation settings.

## Core method

- Primary production method: `DTRR`
- Guarded production scheme: `DTRR + low_flow_guard`
- Fallback method for very low-flow stations: `MAML`

## Code layout

- `code/00_common/`
  - Shared configuration, utilities, model code, and validation wrappers.
- `code/01_Station_Selection/`
  - Station screening and anchor-target partitioning.
- `code/02_Feature_Table/`
  - Static feature construction and hydro-meteorological feature extraction.
- `code/03_Similarity_Matching/`
  - Weighted target-to-anchor similarity matching and donor selection.
- `code/04_Random_30pct_Validation/`
  - Random withholding validation.
- `code/05_Continuous_Gap_Validation/`
  - Continuous gap degradation experiments.
- `code/06_Hybrid_Validation/`
  - Hybrid missingness-scenario validation.
- `code/07_SuperLong_Gap_Analysis/`
  - Validation for gaps longer than 25 months.
- `code/08_GSIM_PLUS_Product/`
  - Final product generation for target and anchor stations.
- `code/09 GRDC_CrossValidation/`
  - GRDC-based external and cross-validation scripts.
- `code/run_all_steps.py`
  - Main pipeline runner.

## Step-by-step workflow

1. `01_Station_Selection/01_station_selector_global.py`
   - Classifies stations into anchor, target, and insufficient groups.
2. `02_Feature_Table/02_build_feature_table_global.py`
   - Builds the main feature table used for similarity analysis.
3. `02_Feature_Table/02_extract_meteo_features_global.py`
   - Extracts meteorological summary features.
4. `02_Feature_Table/02_extract_hydro_features_global.py`
   - Extracts hydrological and basin-related descriptors.
5. `03_Similarity_Matching/03_similarity_matching_global.py`
   - Computes weighted similarity and selects Top-5 donor stations.
6. `04_Random_30pct_Validation/04_random_30pct_validation_noleak.py`
   - Evaluates methods using random 30 % withholding.
7. `05_Continuous_Gap_Validation/05_continuous_gap_validation_noleak.py`
   - Tests performance under continuous missing segments.
8. `06_Hybrid_Validation/06_hybrid_gap_validation_h123_noleak.py`
   - Tests mixed short-gap and long-gap scenarios.
9. `07_SuperLong_Gap_Analysis/07_super_long_gap_analysis.py`
   - Evaluates method robustness on super-long gaps.
10. `08_GSIM_PLUS_Product/08_build_gsim_plus_dataset.py`
    - Builds the final target-station GSIM-PLUS product.
11. `08_GSIM_PLUS_Product/08_build_gsim_plus_anchor_dataset.py`
    - Builds the anchor companion product.

## Supporting modules

- `code/00_common/gsim_plus_config.py`
  - Central path and parameter definitions.
- `code/00_common/gsim_plus_utils.py`
  - Shared data access and helper utilities.
- `code/00_common/gsim_core.py`
  - Core model implementations and DTRR logic.
- `code/00_common/validation_wrappers.py`
  - Method registration and evaluation wrappers.

## External validation

The `code/09 GRDC_CrossValidation/` directory stores the scripts used for GRDC-oriented validation, including:

- global plotting
- USA and Europe station validation
- GRDC matching-table generation
- GRDC validation execution
- four-station comparison utilities

## Figure scripts

The manuscript plotting scripts are stored separately in `fig_code/` and are not required to run the main data-generation pipeline.
