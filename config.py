"""Central configuration. Change settings here, not inside the other files."""
from pathlib import Path

# ---------- folders ----------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"          # put real NASA C-MAPSS files here
DATA_SAMPLE = ROOT / "data" / "sample"    # auto-generated demo data goes here
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# ---------- dataset ----------
DATASET_ID = "FD001"   # C-MAPSS subset: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# The 14 sensors that actually change as an engine degrades (the other 7 are
# constant in FD001, so they carry no information).
FEATURES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

SENSOR_INFO = {
    "s2": "LPC outlet temp", "s3": "HPC outlet temp", "s4": "LPT outlet temp",
    "s7": "HPC outlet pressure", "s8": "Fan speed", "s9": "Core speed",
    "s11": "HPC static pressure", "s12": "Fuel flow / Ps30", "s13": "Corrected fan speed",
    "s14": "Corrected core speed", "s15": "Bypass ratio", "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}

# ---------- modelling ----------
WINDOW_SIZE = 30          # the LSTM looks at the last 30 cycles
RUL_CAP = 125             # RUL above this is treated as "healthy" (standard C-MAPSS practice)
FAILURE_HORIZON = 30      # "failure soon" = RUL <= 30 cycles
VAL_FRACTION = 0.2        # share of training ENGINES held out for validation
LSTM_UNITS = (64, 32)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 40
BATCH_SIZE = 128
SEED = 42

# ---------- maintenance alert thresholds ----------
WARNING_RUL, CRITICAL_RUL = 60, 30
WARNING_PROB, CRITICAL_PROB = 0.40, 0.70
