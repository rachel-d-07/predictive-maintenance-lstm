"""Inference helpers shared by the evaluation script and the Streamlit dashboard."""
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import config as cfg
from src.preprocessing import clean, sliding_windows_padded

MODEL_PATH = cfg.MODELS_DIR / "lstm_rul_model.keras"
SCALER_PATH = cfg.MODELS_DIR / "scaler.joblib"
META_PATH = cfg.MODELS_DIR / "meta.json"


class RULPredictor:
    def __init__(self):
        if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
            raise FileNotFoundError("Trained model not found. Run:  python -m src.train")
        self.model = keras.models.load_model(MODEL_PATH, compile=False)
        self.scaler = joblib.load(SCALER_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.features = self.meta["features"]
        self.window = self.meta["window_size"]
        self.rul_cap = self.meta["rul_cap"]

    def predict_windows(self, raw_windows):
        """raw_windows: (n, window, n_features) in ORIGINAL units -> (rul_cycles, failure_prob)."""
        n, w, f = raw_windows.shape
        scaled = self.scaler.transform(raw_windows.reshape(-1, f)).reshape(n, w, f).astype("float32")
        out = self.model.predict(scaled, batch_size=256, verbose=0)
        rul, fail = (out["rul"], out["failure"]) if isinstance(out, dict) else out
        return np.clip(rul.reshape(-1) * self.rul_cap, 0, self.rul_cap), fail.reshape(-1)

    def predict_window(self, window_data):
        """One machine state from its latest readings (DataFrame or array)."""
        if isinstance(window_data, pd.DataFrame):
            window_data = window_data[self.features].to_numpy(float)
        win = sliding_windows_padded(np.asarray(window_data, float), self.window)[-1][None]
        rul, prob = self.predict_windows(win)
        return float(rul[0]), float(prob[0])

    def predict_series(self, df):
        """Prediction at every cycle of a machine's history."""
        wins = sliding_windows_padded(df[self.features].to_numpy(float), self.window)
        rul, prob = self.predict_windows(wins)
        return pd.DataFrame({"cycle": df["cycle"].to_numpy(), "pred_rul": rul, "failure_prob": prob})


def parse_uploaded(src):
    """Read an uploaded CSV (with headers) or a raw C-MAPSS style .txt file."""
    try:
        df = pd.read_csv(src)
    except Exception:
        df = pd.DataFrame()
    if not set(cfg.FEATURES).issubset(df.columns):
        if hasattr(src, "seek"):
            src.seek(0)
        df2 = pd.read_csv(src, sep=r"\s+", header=None)
        if df2.shape[1] == len(cfg.ALL_COLS):
            df2.columns = cfg.ALL_COLS
            df = df2
    missing = [c for c in cfg.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}. Required columns: {cfg.FEATURES}")
    if "unit" not in df.columns:
        df["unit"] = 1
    if "cycle" not in df.columns:
        df["cycle"] = df.groupby("unit").cumcount() + 1
    return clean(df)
