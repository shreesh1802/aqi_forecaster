"""
model_engine.py
All ML logic: preprocessing, ARIMA, LSTM, hybrid ensemble, spike model.
Mirrors the notebook exactly so results are reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

REQUIRED_COLS = ["CO(GT)", "NO2(GT)", "RH", "T"]
LAG_FEATURES  = ["CO(GT)", "NO2(GT)", "RH", "T"]
TARGET_COL    = "CO(GT)"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, interpolate, add datetime features and lag columns."""
    df = df.copy()

    # Normalise -200 sentinel (AirQualityUCI convention)
    df.replace(-200, np.nan, inplace=True)

    # Parse datetime if present
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce",
        )
        df.drop(columns=["Date", "Time"], inplace=True)
    elif "Datetime" not in df.columns:
        df["Datetime"] = pd.date_range("2004-03-10", periods=len(df), freq="h")

    df = df.sort_values("Datetime").reset_index(drop=True)

    # Keep only numeric cols + Datetime
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[["Datetime"] + num_cols]

    # Forward-fill then back-fill
    df[num_cols] = df[num_cols].ffill().bfill()

    # Temporal features
    df["hour"]  = df["Datetime"].dt.hour
    df["day"]   = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month

    # Lag features
    for col in LAG_FEATURES:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── 2. Sequence helper ────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, target_idx: int, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1
    if mask.sum() == 0:
        return np.inf
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── 3. ARIMA ──────────────────────────────────────────────────────────────────

def fit_arima(train_series: pd.Series):
    from pmdarima import auto_arima
    model = auto_arima(train_series, seasonal=False, error_action="ignore",
                       suppress_warnings=True, stepwise=True)
    return model


# ── 4. LSTM builder ───────────────────────────────────────────────────────────

def build_lstm(input_shape, units: int = 64):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return model


def build_spike_lstm(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="rmsprop", loss="huber")
    return model


# ── 5. Full pipeline ──────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, progress_cb=None) -> dict:
    """
    Run end-to-end hybrid ARIMA-LSTM pipeline.
    progress_cb(step: int, total: int, message: str) — optional Streamlit callback.
    Returns a results dict with predictions, metrics, and raw series.
    """
    total_steps = 9

    def _prog(step, msg):
        if progress_cb:
            progress_cb(step, total_steps, msg)

    # ── Step 1: preprocess ────────────────────────────────────────────────
    _prog(1, "Preprocessing data…")
    df = preprocess(df)

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test  = df.iloc[train_size:].copy()

    train_num = train.drop(columns=["Datetime"])
    test_num  = test.drop(columns=["Datetime"])

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_num)
    test_scaled  = scaler.transform(test_num)
    target_idx   = list(train_num.columns).index(TARGET_COL)

    # ── Step 2: ARIMA ─────────────────────────────────────────────────────
    _prog(2, "Fitting ARIMA model…")
    arima_model = fit_arima(train[TARGET_COL])
    arima_pred  = arima_model.predict(n_periods=len(test))

    # ── Step 3: LSTM hyperparameter search ───────────────────────────────
    _prog(3, "Searching LSTM hyperparameters…")
    best_mape_lstm = np.inf
    best_model     = None
    best_seq       = 24

    for seq_len in [24, 48]:
        for units in [32, 64]:
            X_tr, y_tr = create_sequences(train_scaled, target_idx, seq_len)
            X_te, _    = create_sequences(test_scaled,  target_idx, seq_len)
            m = build_lstm((X_tr.shape[1], X_tr.shape[2]), units)
            m.fit(X_tr, y_tr, epochs=15, batch_size=64, verbose=0)

            ps = m.predict(X_te, verbose=0)
            dummy = np.zeros((len(ps), train_num.shape[1]))
            dummy[:, target_idx] = ps.flatten()
            pred = scaler.inverse_transform(dummy)[:, target_idx]
            actual_tmp = test[TARGET_COL].values[seq_len:][:len(pred)]
            mape = safe_mape(actual_tmp, pred)
            if mape < best_mape_lstm:
                best_mape_lstm = mape
                best_model     = m
                best_seq       = seq_len

    # ── Step 4: Final LSTM predictions ───────────────────────────────────
    _prog(4, "Generating LSTM predictions…")
    X_tr_f, y_tr_f = create_sequences(train_scaled, target_idx, best_seq)
    X_te_f, _      = create_sequences(test_scaled,  target_idx, best_seq)

    lstm_ps = best_model.predict(X_te_f, verbose=0)
    dummy = np.zeros((len(lstm_ps), train_num.shape[1]))
    dummy[:, target_idx] = lstm_ps.flatten()
    lstm_pred = scaler.inverse_transform(dummy)[:, target_idx]

    # ── Step 5: Hybrid weight search ─────────────────────────────────────
    _prog(5, "Optimising hybrid weights…")
    arima_part = arima_pred[best_seq:]
    lstm_part  = lstm_pred
    min_len    = min(len(arima_part), len(lstm_part))
    arima_part = arima_part[:min_len]
    lstm_part  = lstm_part[:min_len]
    actual     = test[TARGET_COL].values[best_seq:][:min_len]

    best_mape_h = np.inf
    best_w      = (0.5, 0.5)
    for w in np.arange(0, 1.01, 0.05):
        hybrid = w * arima_part + (1 - w) * lstm_part
        m_val  = safe_mape(actual, hybrid)
        if m_val < best_mape_h:
            best_mape_h = m_val
            best_w      = (w, 1 - w)

    final_pred = pd.Series(best_w[0] * arima_part + best_w[1] * lstm_part)

    # ── Step 6: Spike model ───────────────────────────────────────────────
    _prog(6, "Training spike detection model…")
    spike_seq = 12
    X_sp, y_sp = create_sequences(train_scaled, target_idx, spike_seq)
    spike_model = build_spike_lstm((X_sp.shape[1], X_sp.shape[2]))
    spike_model.fit(X_sp, y_sp, epochs=20, batch_size=32, verbose=0)

    X_te_sp, _ = create_sequences(test_scaled, target_idx, spike_seq)
    sp_raw     = spike_model.predict(X_te_sp, verbose=0)
    dummy_sp   = np.zeros((len(sp_raw), train_num.shape[1]))
    dummy_sp[:, target_idx] = sp_raw.flatten()
    spike_preds = scaler.inverse_transform(dummy_sp)[:, target_idx]

    # ── Step 7: Ensemble blend ────────────────────────────────────────────
    _prog(7, "Blending ensemble…")
    spike_aligned = spike_preds[-len(final_pred):]
    blend_thresh  = train[TARGET_COL].quantile(0.75)
    combined = []
    for i in range(len(final_pred)):
        if final_pred.iloc[i] > blend_thresh:
            combined.append(0.6 * spike_aligned[i] + 0.4 * final_pred.iloc[i])
        else:
            combined.append(float(final_pred.iloc[i]))
    ensemble_pred = np.array(combined)

    # ── Step 8: Metrics ───────────────────────────────────────────────────
    _prog(8, "Computing metrics…")
    metrics = {
        "MAE":  round(float(mean_absolute_error(actual, ensemble_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, ensemble_pred))), 4),
        "R2":   round(float(r2_score(actual, ensemble_pred)), 4),
        "MAPE": round(float(safe_mape(actual, ensemble_pred)), 2),
        "arima_weight": round(float(best_w[0]), 2),
        "lstm_weight":  round(float(best_w[1]), 2),
        "best_seq":     best_seq,
    }

    # ── Step 9: Build output dataframe ───────────────────────────────────
    _prog(9, "Done.")
    datetimes = test["Datetime"].values[best_seq:][:min_len]
    result_df = pd.DataFrame({
        "Datetime":       datetimes,
        "Actual":         actual,
        "ARIMA":          arima_part,
        "LSTM":           lstm_part,
        "Hybrid":         np.array(final_pred),
        "Ensemble":       ensemble_pred,
        "Residual":       actual - ensemble_pred,
        "Spike_flag":     (actual - ensemble_pred) > train[TARGET_COL].quantile(0.90),
    })

    return {"metrics": metrics, "results": result_df, "df_full": df}
