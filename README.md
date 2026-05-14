# AQI Forecaster — Hybrid ARIMA-LSTM Dashboard

A production-ready Streamlit app for air quality forecasting using a
hybrid ARIMA + LSTM ensemble with dedicated spike detection.

Built as part of the Indian Oil Corporation AQI prediction project.

## Features

- Upload your own CSV (AirQualityUCI format) or run on built-in sample data
- Hybrid ARIMA-LSTM model with optimised weighting
- Dedicated spike model trained on 90th-percentile outlier events
- Interactive Plotly charts: time-series, scatter, residuals, feature correlations
- Model comparison table (ARIMA-only vs LSTM-only vs Ensemble)
- Spike detection tab with event timeline
- Download predictions as CSV

## Project metrics (AirQualityUCI dataset)

| Metric | Value |
|--------|-------|
| R²     | 0.80  |
| MAPE   | ~17%  |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud (free, public URL)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set main file to `app.py`
4. Click Deploy — your app gets a public URL in ~2 minutes

Add the URL to your resume under the project entry.

## Input CSV format

Semicolon-delimited, comma decimal (matches AirQualityUCI.csv):

```
Date;Time;CO(GT);NO2(GT);T;RH
10/03/2004;18.00.00;2,6;166;13,6;48,9
```

- `-200` values are treated as missing and interpolated automatically
- Minimum ~500 rows recommended for reliable ARIMA fitting

## File structure

```
aqi_app/
├── app.py            # Streamlit UI
├── model_engine.py   # All ML logic (ARIMA, LSTM, hybrid, spike)
├── sample_data.py    # Synthetic demo data generator
├── requirements.txt
└── README.md
```
