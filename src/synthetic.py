"""Generador sintético con la misma estructura que la API (modo demo / sin red).

Reproduce los rasgos reales del mercado español: curva de pato por la solar,
picos de mañana y tarde, menor precio en fin de semana, estacionalidad anual,
regímenes persistentes (AR) y peajes del PVPC por periodo (punta/llano/valle).
Es determinista: el mismo instante siempre produce el mismo precio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tolls(local: pd.DatetimeIndex) -> np.ndarray:
    h = local.hour
    weekend = local.dayofweek >= 5
    punta = ((h >= 10) & (h < 14)) | ((h >= 18) & (h < 22))
    llano = ((h >= 8) & (h < 10)) | ((h >= 14) & (h < 18)) | (h >= 22)
    tolls = np.where(punta, 95.0, np.where(llano, 45.0, 8.0))
    return np.where(weekend, 8.0, tolls)


def generate(start: pd.Timestamp, end: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    idx_all = pd.date_range(origin, end.ceil("h"), freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    local = idx_all.tz_convert("Europe/Madrid")
    h = local.hour.values
    doy = local.dayofyear.values

    annual = 70 + 22 * np.cos(2 * np.pi * (doy - 15) / 365)
    solar_depth = 30 + 25 * np.sin(2 * np.pi * (doy - 80) / 365)
    daily = (12 * np.exp(-((h - 8.5) ** 2) / 3)
             + 28 * np.exp(-((h - 21) ** 2) / 4)
             - solar_depth * np.exp(-((h - 14) ** 2) / 8)
             - 10 * np.exp(-((h - 4) ** 2) / 6))
    weekend = np.where(local.dayofweek.values >= 5, -14, 0)
    n_days = len(idx_all) // 24 + 2
    shocks = rng.normal(0, 9, n_days)
    reg = np.zeros(n_days)
    for i in range(1, n_days):
        reg[i] = 0.85 * reg[i - 1] + shocks[i]
    regime = reg[np.arange(len(idx_all)) // 24]
    noise = rng.normal(0, 6, len(idx_all))
    spot = np.clip(annual + daily + weekend + regime + noise, -2, None)

    df = pd.DataFrame({"spot": spot}, index=idx_all)
    df["pvpc"] = df["spot"] * 1.12 + _tolls(local) + 18
    df.index.name = "datetime"
    return df.loc[(df.index >= start) & (df.index <= end)].round(2)
