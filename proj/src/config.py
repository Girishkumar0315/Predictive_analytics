"""Central configuration: paths, random seed and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42          # one seed everywhere -> reproducible results
TEST_SIZE = 0.20           # hold-out fraction (never touched during tuning)
CV_FOLDS = 5               # cross-validation folds for fair model comparison

RAW_FILE = "stackoverflow_survey_single_response.csv"
CROSSWALK_FILE = "qname_levels_single_response_crosswalk.csv"
QUESTIONS_FILE = "stackoverflow_survey_questions.csv"

for _d in (DATA_PROC, FIG_DIR, RES_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
