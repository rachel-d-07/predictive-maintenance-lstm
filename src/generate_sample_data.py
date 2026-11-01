"""Creates SYNTHETIC run-to-failure sensor data in the same file format as NASA C-MAPSS.

It is only a stand-in so the project runs without downloading anything.
For your real submission, put the real C-MAPSS files in data/raw/ (see README).
Run:  python -m src.generate_sample_data
"""
import numpy as np
import pandas as pd
import config as cfg

# sensor: (healthy baseline, total drift by end of life, noise std)
PROFILE = {
    "s2": (641.8, 1.8, 0.45),   "s3": (1580.0, 30.0, 6.0),   "s4": (1395.0, 40.0, 8.0),
    "s7": (554.5, -2.2, 0.5),   "s8": (2388.0, 0.45, 0.07),  "s9": (9045.0, 45.0, 12.0),
    "s11": (46.9, 1.5, 0.25),   "s12": (522.5, -3.5, 0.35),  "s13": (2388.0, 0.4, 0.07),
    "s14": (8130.0, 40.0, 9.0), "s15": (8.40, 0.2, 0.037),   "s17": (391.0, 7.0, 1.5),
    "s20": (39.1, -0.9, 0.15),  "s21": (23.45, -0.5, 0.09),
}
CONSTANTS = {"s1": 518.67, "s5": 14.62, "s6": 21.61, "s10": 1.30,
             "s16": 0.03, "s18": 2388.0, "s19": 100.0}


def simulate_engine(unit, life, rng, n_cycles=None):
    """Simulate one engine. Wear grows slowly at first, then quickly near failure."""
    n = n_cycles or life
    t = np.arange(1, n + 1)
    wear = (t / life) ** 2.2 * rng.uniform(0.85, 1.15)
    data = {"unit": np.full(n, unit), "cycle": t,
            "setting_1": rng.normal(0, 0.002, n), "setting_2": rng.normal(0, 0.0003, n),
            "setting_3": np.full(n, 100.0)}
    for s in cfg.SENSOR_COLS:
        if s in PROFILE:
            base, drift, noise = PROFILE[s]
            data[s] = base + rng.normal(0, 0.8 * noise) + drift * wear + rng.normal(0, noise, n)
        else:
            data[s] = np.full(n, CONSTANTS[s])
    return pd.DataFrame(data)[cfg.ALL_COLS]


def _save_cmapss(df, path):
    df.to_csv(path, sep=" ", header=False, index=False, float_format="%.4f")


def generate_all(n_train=100, n_test=100, seed=cfg.SEED):
    cfg.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lifes = lambda n: np.clip(rng.normal(205, 40, n).astype(int), 130, 360)

    train = pd.concat([simulate_engine(u, L, rng) for u, L in enumerate(lifes(n_train), 1)])

    test_parts, rul = [], []
    for u, L in enumerate(lifes(n_test), 1):
        cut = int(rng.integers(40, L - 5))          # we only observe the first `cut` cycles
        test_parts.append(simulate_engine(u, L, rng, n_cycles=cut))
        rul.append(L - cut)                          # true remaining life (ground truth)
    test = pd.concat(test_parts)

    d = cfg.DATASET_ID
    _save_cmapss(train, cfg.DATA_SAMPLE / f"train_{d}.txt")
    _save_cmapss(test, cfg.DATA_SAMPLE / f"test_{d}.txt")
    pd.Series(rul).to_csv(cfg.DATA_SAMPLE / f"RUL_{d}.txt", index=False, header=False)

    # three small CSVs you can upload in the dashboard
    demo_rng = np.random.default_rng(7)
    for name, cycles in [("healthy", 70), ("degrading", 200), ("critical", 242)]:
        simulate_engine(1, 250, demo_rng, n_cycles=cycles).to_csv(
            cfg.DATA_SAMPLE / f"demo_{name}_engine.csv", index=False)
    print(f"Synthetic sample data written to {cfg.DATA_SAMPLE}")


if __name__ == "__main__":
    generate_all()
