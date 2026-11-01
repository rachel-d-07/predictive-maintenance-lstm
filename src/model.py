"""Two-output LSTM: (1) RUL regression, (2) probability of failure within the horizon."""
from tensorflow import keras
from tensorflow.keras import layers
import config as cfg


def build_model(window, n_features):
    inp = keras.Input(shape=(window, n_features), name="sensor_window")
    x = layers.LSTM(cfg.LSTM_UNITS[0], return_sequences=True)(inp)
    x = layers.Dropout(cfg.DROPOUT)(x)
    x = layers.LSTM(cfg.LSTM_UNITS[1])(x)
    x = layers.Dropout(cfg.DROPOUT)(x)
    x = layers.Dense(32, activation="relu")(x)
    rul = layers.Dense(1, name="rul")(x)                         # RUL / RUL_CAP
    fail = layers.Dense(1, activation="sigmoid", name="failure")(x)
    model = keras.Model(inp, [rul, fail])
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.LEARNING_RATE),
        loss={"rul": "mse", "failure": "binary_crossentropy"},
        loss_weights={"rul": 1.0, "failure": 0.5},
    )
    return model
