# config/settings.py
# Purpose: Central configuration — paths, constants, auto-creation of directories.

from pathlib import Path

# ----- ROOT -----
ROOT_DIR = Path(__file__).resolve().parent.parent

# ----- DATA -----
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "uploads"             # uploaded nomenclature files
REFERENCE_DIR = DATA_DIR / "reference"          # okpd_2.xlsx, abbreviations.xlsx, vat_exempt_codes.xlsx
OUTPUT_DIR = DATA_DIR / "output"               # versioned results
TRAINING_DATA_DIR = DATA_DIR / "uploads"

# ----- ARTIFACTS -----
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FAISS_DIR = ARTIFACTS_DIR / "faiss"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# ----- FRONTEND -----
FRONTEND_DIR = ROOT_DIR / "frontend"

# ----- ENSURE DIRS EXIST -----
for directory in [
    RAW_DATA_DIR, REFERENCE_DIR, OUTPUT_DIR,
    MODELS_DIR, FAISS_DIR, LOGS_DIR,
    FRONTEND_DIR, TRAINING_DATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ----- DEFAULT SETTINGS -----
DEFAULT_SETTINGS = {
    "auto_threshold": 0.85,
    "review_threshold": 0.40,
    "ood_score_threshold": 0.4,
    "ood_entropy_threshold": 2.5,
    "enable_hierarchy": True,
    "enable_ner_boosting": True,
    "enable_ood": True,
    "enable_ensemble": False,
    "active_model": "embeddinggemma-300m",
    "reranker_model": "USER-bge-m3",
    "mrl_dimension": 256,
    "taxonomy_sections": "A,C",
    "inference_blocked": False,
}