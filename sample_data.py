"""
sample_data.py
Generates a realistic synthetic AirQualityUCI-format CSV for demo use.
"""

import numpy as np
import pandas as pd
import io


def generate_sample_csv(n_hours: int = 1500) -> bytes:
    """Return bytes of a CSV that mimics the AirQualityUCI dataset."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2004-03-10 18:00:00", periods=n_hours, freq="h")

    # Seasonal + daily cycles for realism
    hour_arr  = dates.hour.values
    month_arr = dates.month.values

    # CO(GT) — target
    trend    = np.linspace(2.5, 1.5, n_hours)
    daily    = 1.2 * np.sin(2 * np.pi * (hour_arr - 7) / 24)
    seasonal = 0.8 * np.sin(2 * np.pi * month_arr / 12)
    noise    = rng.normal(0, 0.4, n_hours)
    spikes   = rng.choice([0, 4], size=n_hours, p=[0.97, 0.03])
    co = np.clip(trend + daily + seasonal + noise + spikes, 0.1, 12)

    # NO2(GT)
    no2 = np.clip(
        80 + 30 * np.sin(2 * np.pi * (hour_arr - 8) / 24)
        + 10 * np.sin(2 * np.pi * month_arr / 12)
        + rng.normal(0, 8, n_hours),
        5, 300,
    )

    # Temperature (°C)
    temp = np.clip(
        15 + 8 * np.sin(2 * np.pi * (hour_arr - 14) / 24)
        + 5 * np.sin(2 * np.pi * (month_arr - 7) / 12)
        + rng.normal(0, 1.5, n_hours),
        -5, 40,
    )

    # Relative Humidity (%)
    rh = np.clip(
        60 - 0.5 * (temp - 15) + rng.normal(0, 5, n_hours),
        20, 95,
    )

    df = pd.DataFrame({
        "Date":    dates.strftime("%d/%m/%Y"),
        "Time":    dates.strftime("%H.%M.%S"),
        "CO(GT)":  np.round(co, 2),
        "NO2(GT)": np.round(no2, 1),
        "T":       np.round(temp, 1),
        "RH":      np.round(rh, 1),
    })

    # Sprinkle ~3 % missing values as -200 (UCI convention)
    for col in ["CO(GT)", "NO2(GT)", "T", "RH"]:
        mask = rng.random(n_hours) < 0.03
        df.loc[mask, col] = -200

    buf = io.BytesIO()
    df.to_csv(buf, index=False, sep=";", decimal=",")
    return buf.getvalue()
