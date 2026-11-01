"""Real evaluation on the held-out test engines + all graphs. Called by train.py."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, precision_score,
                             recall_score, f1_score, accuracy_score, confusion_matrix)
import config as cfg


def nasa_score(y_true, y_pred):
    """Asymmetric C-MAPSS score: late predictions are punished harder than early ones. Lower = better."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def score_test_engines(predictor, test_df, rul_test):
    """Predict RUL at every cycle of each test engine; keep the final cycle for the metrics."""
    rows, series = [], {}
    for unit, g in test_df.groupby("unit"):
        g = g.sort_values("cycle")
        res = predictor.predict_series(g)
        last_cycle, true_raw = int(g["cycle"].max()), float(rul_test.loc[unit])
        res["true_rul"] = np.minimum(true_raw + (last_cycle - res["cycle"].to_numpy()), cfg.RUL_CAP)
        series[unit] = res
        rows.append({"unit": unit, "last_cycle": last_cycle, "true_rul_raw": true_raw,
                     "true_rul": min(true_raw, cfg.RUL_CAP),
                     "pred_rul": res["pred_rul"].iloc[-1], "failure_prob": res["failure_prob"].iloc[-1]})
    return pd.DataFrame(rows), series


def compute_metrics(final, train_df, source):
    y, p = final["true_rul"].to_numpy(), final["pred_rul"].to_numpy()
    true_fail = (final["true_rul_raw"] <= cfg.FAILURE_HORIZON).astype(int).to_numpy()
    pred_fail = (final["failure_prob"] >= 0.5).astype(int).to_numpy()
    base = np.full_like(y, train_df["rul"].mean())          # naive "always predict the average" baseline
    return {
        "data_source": source,
        "n_test_engines": int(len(final)),
        "rul": {"mae": float(mean_absolute_error(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "r2": float(r2_score(y, p)), "nasa_score": nasa_score(y, p)},
        "baseline_mean_predictor": {"mae": float(mean_absolute_error(y, base)),
                                    "rmse": float(np.sqrt(mean_squared_error(y, base)))},
        "failure_classification": {
            "horizon_cycles": cfg.FAILURE_HORIZON, "threshold": 0.5,
            "precision": float(precision_score(true_fail, pred_fail, zero_division=0)),
            "recall": float(recall_score(true_fail, pred_fail, zero_division=0)),
            "f1": float(f1_score(true_fail, pred_fail, zero_division=0)),
            "accuracy": float(accuracy_score(true_fail, pred_fail)),
            "n_failing_engines": int(true_fail.sum()),
            "confusion_matrix": confusion_matrix(true_fail, pred_fail, labels=[0, 1]).tolist()},
    }


def make_plots(history, final, series, metrics):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    y, p = final["true_rul"].to_numpy(), final["pred_rul"].to_numpy()

    # 1. training curves
    plt.figure(figsize=(7, 4))
    plt.plot(history["loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training curves"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "training_curves.png", dpi=130); plt.close()

    # 2. actual vs predicted scatter
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y, p, alpha=.7)
    lim = [0, cfg.RUL_CAP + 5]
    plt.plot(lim, lim, "r--", label="perfect prediction")
    plt.xlabel("Actual RUL (cycles)"); plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"Actual vs predicted RUL (MAE {metrics['rul']['mae']:.1f}, RMSE {metrics['rul']['rmse']:.1f})")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "actual_vs_predicted_scatter.png", dpi=130); plt.close()

    # 3. actual vs predicted per engine (sorted by true RUL)
    order = np.argsort(y)
    plt.figure(figsize=(10, 4))
    plt.plot(y[order], "o-", label="Actual RUL", ms=3)
    plt.plot(p[order], "s-", label="Predicted RUL", ms=3)
    plt.xlabel("Test engines (sorted by actual RUL)"); plt.ylabel("RUL (cycles)")
    plt.title("Actual vs predicted RUL for every test engine"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(cfg.FIG_DIR / "actual_vs_predicted_by_engine.png", dpi=130); plt.close()

    # 4. RUL trajectories of 4 engines with the longest observed history
    longest = final.sort_values("last_cycle", ascending=False)["unit"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, u in zip(axes.ravel(), longest):
        r = series[u]
        ax.plot(r["cycle"], r["true_rul"], "--", label="Actual RUL")
        ax.plot(r["cycle"], r["pred_rul"], label="Predicted RUL")
        ax.set_title(f"Engine {u}"); ax.grid(alpha=.3)
    axes[0, 0].legend(); fig.supxlabel("Cycle"); fig.supylabel("RUL (cycles)")
    fig.suptitle("RUL prediction over each engine's life"); fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "rul_trajectories.png", dpi=130); plt.close(fig)

    # 5. error histogram
    plt.figure(figsize=(6, 4))
    plt.hist(p - y, bins=20, edgecolor="black")
    plt.axvline(0, color="r", ls="--")
    plt.xlabel("Prediction error (predicted - actual, cycles)"); plt.ylabel("Engines")
    plt.title("Error distribution"); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "error_histogram.png", dpi=130); plt.close()

    # 6. confusion matrix
    cm = np.array(metrics["failure_classification"]["confusion_matrix"])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=16)
    plt.xticks([0, 1], ["Not failing", "Failing soon"]); plt.yticks([0, 1], ["Not failing", "Failing soon"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Failure within {cfg.FAILURE_HORIZON} cycles"); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "confusion_matrix.png", dpi=130); plt.close()


def run_full_evaluation(history, train_df, test_df, rul_test, source, predictor):
    final, series = score_test_engines(predictor, test_df, rul_test)
    metrics = compute_metrics(final, train_df, source)
    make_plots(history, final, series, metrics)
    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    (cfg.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    final.to_csv(cfg.REPORTS_DIR / "test_predictions.csv", index=False)
    return metrics
