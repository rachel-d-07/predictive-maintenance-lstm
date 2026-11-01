"""Loads NASA C-MAPSS from data/raw/ if present, otherwise the synthetic sample."""
import pandas as pd
import config as cfg


def read_cmapss(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=cfg.ALL_COLS)


def _find_source():
    needed = [f"train_{cfg.DATASET_ID}.txt", f"test_{cfg.DATASET_ID}.txt", f"RUL_{cfg.DATASET_ID}.txt"]
    for folder, label in [(cfg.DATA_RAW, "NASA C-MAPSS (real data)"),
                          (cfg.DATA_SAMPLE, "Synthetic sample data (generated)")]:
        if all((folder / f).exists() for f in needed):
            return folder, label
    return None, None


def get_dataset():
    """Returns train_df, test_df, rul_test (Series indexed by unit), source_label."""
    folder, label = _find_source()
    if folder is None:
        from src.generate_sample_data import generate_all
        generate_all()
        folder, label = _find_source()
    d = cfg.DATASET_ID
    train = read_cmapss(folder / f"train_{d}.txt")
    test = read_cmapss(folder / f"test_{d}.txt")
    rul_values = pd.read_csv(folder / f"RUL_{d}.txt", sep=r"\s+", header=None)[0].values
    rul_test = pd.Series(rul_values, index=sorted(test["unit"].unique()), name="rul")
    return train, test, rul_test, label
