"""Ingeniería de variables para predecir el PVPC a 48 h vista.

Todas las variables dependientes del precio usan retardos >= 48 h, de modo que
un único modelo directo puede predecir las 48 horas siguientes al último dato
conocido sin realimentar sus propias predicciones (sin fuga de información).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

import config

LAGS = [48, 72, 96, 168, 336]

FEATURE_LABELS = {
    "hour": "Hora del día", "dow": "Día de la semana", "month": "Mes",
    "is_weekend": "Fin de semana", "is_holiday": "Festivo nacional",
    "hour_sin": "Hora (seno)", "hour_cos": "Hora (coseno)",
    "doy_sin": "Día del año (seno)", "doy_cos": "Día del año (coseno)",
    "lag_48": "Precio hace 48 h", "lag_72": "Precio hace 72 h",
    "lag_96": "Precio hace 96 h", "lag_168": "Precio hace 1 semana",
    "lag_336": "Precio hace 2 semanas", "mean24_48": "Media diaria hace 2 días",
    "mean168_48": "Media semanal (retrasada 48 h)", "std24_48": "Volatilidad diaria (retrasada)",
    "spot_lag_48": "Mercado mayorista hace 48 h",
}


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    local = idx.tz_convert(config.TIMEZONE)
    es_holidays = holidays.Spain(years=range(local.year.min(), local.year.max() + 1))
    df = pd.DataFrame(index=idx)
    df["hour"] = local.hour
    df["dow"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df["is_holiday"] = [int(d in es_holidays) for d in local.date]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    doy = local.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def build_features(prices: pd.DataFrame, extend_hours: int = 0) -> pd.DataFrame:
    """Devuelve matriz de variables (+ target 'y') sobre una rejilla horaria
    continua, ampliada extend_hours por delante del último dato para predecir."""
    s = prices["pvpc"].asfreq("1h")
    if extend_hours:
        future = pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=extend_hours,
                               freq="1h", tz="UTC")
        s = pd.concat([s, pd.Series(np.nan, index=future)])
    spot = prices["spot"].reindex(s.index)

    X = calendar_features(s.index)
    for lag in LAGS:
        X[f"lag_{lag}"] = s.shift(lag)
    X["mean24_48"] = s.shift(48).rolling(24, min_periods=18).mean()
    X["mean168_48"] = s.shift(48).rolling(168, min_periods=120).mean()
    X["std24_48"] = s.shift(48).rolling(24, min_periods=18).std()
    X["spot_lag_48"] = spot.shift(48)
    X["y"] = s
    return X


FEATURES = [c for c in FEATURE_LABELS]
