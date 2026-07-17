import os
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", REPOSITORY_DIR)).expanduser().resolve()
LEGACY_DIR = Path(os.environ.get("GSIM_PLUS_LEGACY_DIR", PROJECT_DIR / "legacy")).expanduser().resolve()
LEGACY_SCRIPTS_DIR = LEGACY_DIR / "scripts"
PROJECT_MATERIAL_DIR = Path(
    os.environ.get("GSIM_PLUS_MATERIAL_DIR", PROJECT_DIR / "999 material")
).expanduser().resolve()
ASCII_MATERIAL_VALUE = os.environ.get("GSIM_PLUS_ASCII_MATERIAL_DIR")
ASCII_MATERIAL_DIR = (
    Path(ASCII_MATERIAL_VALUE).expanduser().resolve() if ASCII_MATERIAL_VALUE else None
)
MATERIAL_DIR = ASCII_MATERIAL_DIR or PROJECT_MATERIAL_DIR

MONTHLY_DIR = Path(
    os.environ.get(
        "GSIM_MONTHLY_DIR",
        PROJECT_DIR / "inputs" / "GSIM_indices" / "TIMESERIES" / "monthly",
    )
).expanduser().resolve()
GLOBAL_ATTR_FILE = MATERIAL_DIR / "GSIM_attribute.csv"
CLIMATE_FILE = MATERIAL_DIR / "stations_full_static_with_climate_kg_feat.csv"
METEO_NC = MATERIAL_DIR / "Global_1995-2015.nc"
ATTRIBUTE_DIR = MATERIAL_DIR / "Attribute"
HYDRORIVERS_SHP = ATTRIBUTE_DIR / "River" / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"
HYBAS_DIR = ATTRIBUTE_DIR / "BASIN"

STEP1_DIR = PROJECT_DIR / "01_Station_Selection"
STEP2_DIR = PROJECT_DIR / "02_Feature_Table"
STEP3_DIR = PROJECT_DIR / "03_Similarity_Matching"
STEP4_DIR = PROJECT_DIR / "04_Random_30pct_Validation"
STEP5_DIR = PROJECT_DIR / "05_Continuous_Gap_Validation"
STEP6_DIR = PROJECT_DIR / "06_Hybrid_Validation"
STEP7_DIR = PROJECT_DIR / "07_SuperLong_Gap_Analysis"
STEP8_DIR = PROJECT_DIR / "08_GSIM_PLUS_Product"
MODEL_CACHE_DIR = PROJECT_DIR / "00_common" / "model_cache"

STUDY_START_YEAR = 1995
STUDY_END_YEAR = 2015
STUDY_YEARS = list(range(STUDY_START_YEAR, STUDY_END_YEAR + 1))
N_STUDY_MONTHS = 12 * (STUDY_END_YEAR - STUDY_START_YEAR + 1)

ANCHOR_THRESHOLD = 0.90
TARGET_MIN = 0.30
TARGET_MAX = 0.90
EVALUABLE_MIN_MONTHS = 120

RANDOM_SEED = 42
K_NEIGHBORS = 5

H1_WEIGHTS = {
    "1": 0.35,
    "2-3": 0.30,
    "4-6": 0.20,
    "7-12": 0.10,
    "13-24": 0.05,
    "25+": 0.00,
}

H2_WEIGHTS = {
    "1": 0.20,
    "2-3": 0.20,
    "4-6": 0.20,
    "7-12": 0.15,
    "13-24": 0.15,
    "25+": 0.10,
}

H3_WEIGHTS = {
    "1": 0.10,
    "2-3": 0.10,
    "4-6": 0.15,
    "7-12": 0.20,
    "13-24": 0.20,
    "25+": 0.25,
}

SUPER_LONG_MIN_MONTHS = 25

for path in [
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    STEP4_DIR,
    STEP5_DIR,
    STEP6_DIR,
    STEP7_DIR,
    STEP8_DIR,
    MODEL_CACHE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
