"""Cleaning, RUL labels, normalisation and time-series window generation."""
import numpy as np
import pandas as pd
import config as cfg


def clean(df):
    """Remove duplicates, fix missing/non-numeric values (per engine)."""
    df = df.drop_duplicates(subset=["unit", "cycle"]).sort_values(["unit", "cycle"]).reset_index(drop=True)
    df[cfg.FEATURES] = df[cfg.FEATURES].apply(pd.to_numeric, errors="coerce")
    df[cfg.FEATURES] = df.groupby("unit")[cfg.FEATURES].transform(
        lambda s: s.interpolate(limit_direction="both"))
    df[cfg.FEATURES] = df[cfg.FEATURES].fillna(df[cfg.FEATURES].median())
    return df


def add_rul(df):
    """RUL = cycles left until the engine's last recorded cycle (its failure)."""
    df = df.copy()
    df["rul_raw"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul"] = df["rul_raw"].clip(upper=cfg.RUL_CAP)   # piece-wise linear RUL target
    return df


def split_units(units, val_fraction=cfg.VAL_FRACTION, seed=cfg.SEED):
    """Split by ENGINE (not by row) so validation engines are completely unseen."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    return units[n_val:], units[:n_val]


def sliding_windows_padded(arr, window=cfg.WINDOW_SIZE):
    """One window per row (window ends at that row). Early rows are padded by repeating row 0."""
    pad = np.repeat(arr[:1], window - 1, axis=0)
    padded = np.vstack([pad, arr])
    return np.stack([padded[i:i + window] for i in range(len(arr))])


def make_windows(df, scaler):
    """Build (X, y_rul, y_fail) training arrays from full-length windows."""
    X, y_rul, y_fail = [], [], []
    for _, g in df.groupby("unit"):
        arr = scaler.transform(g[cfg.FEATURES].to_numpy(float))
        rul, raw = g["rul"].to_numpy(), g["rul_raw"].to_numpy()
        for end in range(cfg.WINDOW_SIZE, len(g) + 1):
            X.append(arr[end - cfg.WINDOW_SIZE:end])
            y_rul.append(rul[end - 1] / cfg.RUL_CAP)                  # scaled to 0..1
            y_fail.append(float(raw[end - 1] <= cfg.FAILURE_HORIZON))  # 1 = fails within horizon
    return (np.asarray(X, "float32"), np.asarray(y_rul, "float32").reshape(-1, 1),
            np.asarray(y_fail, "float32").reshape(-1, 1))
