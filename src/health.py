"""Turns model outputs into a health status, risk %, and maintenance alerts."""
import numpy as np
import config as cfg


def health_report(rul, failure_prob):
    risk_pct = float(failure_prob) * 100
    health_score = float(100 * np.clip(rul, 0, cfg.RUL_CAP) / cfg.RUL_CAP)

    if rul <= cfg.CRITICAL_RUL or failure_prob >= cfg.CRITICAL_PROB:
        status = "CRITICAL"
        alerts = [f"🔴 CRITICAL: failure expected in about {rul:.0f} cycles. Stop and inspect / schedule maintenance now."]
    elif rul <= cfg.WARNING_RUL or failure_prob >= cfg.WARNING_PROB:
        status = "WARNING"
        alerts = [f"🟠 WARNING: health is degrading (~{rul:.0f} cycles left). Plan maintenance in the next service window."]
    else:
        status = "HEALTHY"
        alerts = ["🟢 Normal operation. No maintenance action needed."]

    if failure_prob >= cfg.WARNING_PROB:
        alerts.append(f"⚠️ Probability of failure within {cfg.FAILURE_HORIZON} cycles: {risk_pct:.0f}%.")
    return {"status": status, "risk_pct": risk_pct, "health_score": health_score,
            "rul": float(rul), "alerts": alerts}


def trend_alert(prob_series, lookback=10, jump=0.20):
    """Extra alert if the failure probability is rising fast."""
    p = np.asarray(prob_series)
    if len(p) > lookback and p[-1] - p[-1 - lookback] >= jump:
        return f"📈 Failure risk rose by {100 * (p[-1] - p[-1 - lookback]):.0f} points in the last {lookback} cycles."
    return None


def degradation_profile(train_df):
    """Per-sensor std and the typical healthy -> end-of-life shift (used by the simulator)."""
    std = train_df[cfg.FEATURES].std().to_numpy()
    first = train_df.groupby("unit").head(20)[cfg.FEATURES].mean().to_numpy()
    last = train_df.groupby("unit").tail(10)[cfg.FEATURES].mean().to_numpy()
    return std, last - first
