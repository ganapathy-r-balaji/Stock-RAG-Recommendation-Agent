"""
Forecasting module — LightGBM 3-day-ahead price predictor.

Feature engineering → walk-forward validation → train → predict.
Exposed as a single function: forecast_price(ticker) → ForecastResult.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.data_layer import get_price_history


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    ticker: str
    current_price: float
    forecast_price: float          # 3-day-ahead prediction
    forecast_pct_change: float     # % change vs current
    confidence: float              # 0-1; based on walk-forward MAE vs price scale
    direction: str                 # "up" | "down" | "flat"


# ── Feature engineering ────────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # Lagged closes
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = close.shift(lag)

    # Rolling statistics
    for w in [5, 10, 20]:
        df[f"ma_{w}"]  = close.rolling(w).mean()
        df[f"std_{w}"] = close.rolling(w).std()

    # RSI (14)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume change
    df["vol_change"] = df["volume"].pct_change()

    # Target: close 3 days ahead
    df["target"] = close.shift(-3)

    return df.dropna()


# ── Walk-forward backtesting / training ───────────────────────────────────────

_FEATURE_COLS = [
    "lag_1","lag_2","lag_3","lag_5","lag_10",
    "ma_5","ma_10","ma_20",
    "std_5","std_10","std_20",
    "rsi_14","vol_change",
]

def _walk_forward_mae(df: pd.DataFrame, n_splits: int = 5) -> float:
    """Return average MAE across walk-forward folds."""
    fold_size = len(df) // (n_splits + 1)
    maes = []
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error
    for i in range(1, n_splits + 1):
        train = df.iloc[:fold_size * i]
        test  = df.iloc[fold_size * i: fold_size * (i + 1)]
        if len(test) == 0:
            continue
        model = LGBMRegressor(n_estimators=100, verbose=-1)
        model.fit(train[_FEATURE_COLS], train["target"])
        preds = model.predict(test[_FEATURE_COLS])
        maes.append(mean_absolute_error(test["target"], preds))
    return float(np.mean(maes)) if maes else float("inf")


# ── Public API ─────────────────────────────────────────────────────────────────

def forecast_price(ticker: str) -> ForecastResult:
    """
    Fetch price history, engineer features, train LightGBM, and return a
    3-day-ahead forecast with a confidence score.
    """
    raw_df = get_price_history(ticker, days=365)
    df     = _add_features(raw_df)

    if len(df) < 30:
        raise ValueError(f"Not enough price history for {ticker!r} to build a forecast.")

    # Walk-forward MAE → confidence
    mae        = _walk_forward_mae(df)
    price_scale = df["close"].mean()
    # Confidence: 1 − (MAE / price_scale), floored at 0, capped at 1
    confidence = float(np.clip(1 - mae / price_scale, 0.0, 1.0))

    # Final model trained on all data
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=200, verbose=-1)
    model.fit(df[_FEATURE_COLS], df["target"])

    # Predict from the last row (most recent candle)
    last_row   = df[_FEATURE_COLS].iloc[[-1]]
    prediction = float(model.predict(last_row)[0])
    current    = float(df["close"].iloc[-1])
    pct_change = (prediction - current) / current * 100

    direction = "flat"
    if pct_change > 1.0:
        direction = "up"
    elif pct_change < -1.0:
        direction = "down"

    return ForecastResult(
        ticker=ticker,
        current_price=round(current, 2),
        forecast_price=round(prediction, 2),
        forecast_pct_change=round(pct_change, 2),
        confidence=round(confidence, 3),
        direction=direction,
    )
