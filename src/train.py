"""Train the LSTM and evaluate it.
Run:  python -m src.train                       (defaults)
      python -m src.train --epochs 100 --patience 15
"""
import argparse
import json
from datetime import datetime
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import config as cfg
from src.data_loader import get_dataset
from src.preprocessing import clean, add_rul, split_units, make_windows
from src.model import build_model
from src import predict, evaluate


def main(epochs, patience):
    keras.utils.set_random_seed(cfg.SEED)
    cfg.MODELS_DIR.mkdir(exist_ok=True)

    print("1/5  Loading data ...")
    train_raw, test_raw, rul_test, source = get_dataset()
    print(f"     source: {source}")
    train_df = add_rul(clean(train_raw))
    test_df = clean(test_raw)

    print("2/5  Splitting by engine, scaling, building windows ...")
    train_units, val_units = split_units(train_df["unit"].unique())
    tr = train_df[train_df["unit"].isin(train_units)]
    va = train_df[train_df["unit"].isin(val_units)]
    scaler = MinMaxScaler().fit(tr[cfg.FEATURES].to_numpy(float))     # fit on TRAIN engines only
    X_tr, yr_tr, yf_tr = make_windows(tr, scaler)
    X_va, yr_va, yf_va = make_windows(va, scaler)
    print(f"     train windows {X_tr.shape}, validation windows {X_va.shape}")

    print("3/5  Training LSTM ...")
    model = build_model(cfg.WINDOW_SIZE, len(cfg.FEATURES))
    model.summary()
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)]
    hist = model.fit(X_tr, {"rul": yr_tr, "failure": yf_tr},
                     validation_data=(X_va, {"rul": yr_va, "failure": yf_va}),
                     epochs=epochs, batch_size=cfg.BATCH_SIZE, callbacks=callbacks, verbose=2)

    print("4/5  Saving model ...")
    model.save(predict.MODEL_PATH)
    joblib.dump(scaler, predict.SCALER_PATH)
    predict.META_PATH.write_text(json.dumps({
        "features": cfg.FEATURES, "window_size": cfg.WINDOW_SIZE, "rul_cap": cfg.RUL_CAP,
        "failure_horizon": cfg.FAILURE_HORIZON, "data_source": source,
        "epochs_run": len(hist.history["loss"]), "trained_at": datetime.now().isoformat(timespec="seconds")}, indent=2))

    print("5/5  Evaluating on the held-out test engines ...")
    predictor = predict.RULPredictor()
    m = evaluate.run_full_evaluation(hist.history, train_df, test_df, rul_test, source, predictor)

    r, c, b = m["rul"], m["failure_classification"], m["baseline_mean_predictor"]
    print("\n================ REAL TEST RESULTS ================")
    print(f"Data source      : {source}")
    print(f"Epochs run       : {len(hist.history['loss'])}")
    print(f"RUL  MAE         : {r['mae']:.2f} cycles   (baseline 'always predict average': {b['mae']:.2f})")
    print(f"RUL  RMSE        : {r['rmse']:.2f} cycles   (baseline: {b['rmse']:.2f})")
    print(f"RUL  R2          : {r['r2']:.3f}")
    print(f"NASA score       : {r['nasa_score']:.1f}  (lower is better)")
    print(f"Failure<= {c['horizon_cycles']} cycles: precision {c['precision']:.3f} | recall {c['recall']:.3f} | F1 {c['f1']:.3f}")
    print(f"Confusion matrix : {c['confusion_matrix']}")
    print("Graphs saved to reports/figures/ , metrics to reports/metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop if validation loss hasn't improved for this many epochs")
    args = ap.parse_args()
    main(args.epochs, args.patience)