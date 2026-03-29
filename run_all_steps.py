import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    PROJECT_DIR / "01_Station_Selection" / "01_station_selector_global.py",
    PROJECT_DIR / "02_Feature_Table" / "02_build_feature_table_global.py",
    PROJECT_DIR / "02_Feature_Table" / "02_extract_meteo_features_global.py",
    PROJECT_DIR / "03_Similarity_Matching" / "03_similarity_matching_global.py",
    PROJECT_DIR / "04_Random_30pct_Validation" / "04_random_30pct_validation_noleak.py",
    PROJECT_DIR / "05_Continuous_Gap_Validation" / "05_continuous_gap_validation_noleak.py",
    PROJECT_DIR / "06_Hybrid_Validation" / "06_hybrid_gap_validation_h123_noleak.py",
    PROJECT_DIR / "07_SuperLong_Gap_Analysis" / "07_super_long_gap_analysis.py",
    PROJECT_DIR / "08_GSIM_PLUS_Product" / "08_build_gsim_plus_dataset.py",
]


def main():
    for script in SCRIPTS:
        print(f"\nRunning: {script.name}")
        result = subprocess.run(["python", str(script)], cwd=str(PROJECT_DIR))
        if result.returncode != 0:
            raise SystemExit(f"Step failed: {script}")


if __name__ == "__main__":
    main()
