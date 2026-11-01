"""Create demo upload files from REAL C-MAPSS test engines.
Run:  python -m src.make_demo_files
"""
import pandas as pd
import config as cfg
from src.data_loader import get_dataset


def closest(df, target):
    """Row whose true remaining life is closest to `target` cycles."""
    return df.loc[(df["true_rul_raw"] - target).abs().idxmin()]


def main():
    _, test, _, source = get_dataset()
    if "real" not in source.lower():
        raise SystemExit("Real C-MAPSS files not found in data/raw/. Aborting.")
    preds = pd.read_csv(cfg.REPORTS_DIR / "test_predictions.csv")
    preds = preds[preds["last_cycle"] >= 60]          # need some history

    for name, target in [("healthy", 110), ("degrading", 60), ("critical", 8)]:
        row = closest(preds, target)
        unit = int(row["unit"])
        out = cfg.DATA_SAMPLE / f"demo_{name}_engine.csv"
        test[test["unit"] == unit].to_csv(out, index=False)
        print(f"{name:10s}: engine {unit}, {int(row['last_cycle'])} cycles, "
              f"true RUL at last cycle = {row['true_rul_raw']:.0f}  ->  {out.name}")


if __name__ == "__main__":
    main()