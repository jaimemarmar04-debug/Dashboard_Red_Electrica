"""Entrenamiento, evaluación y predicción del modelo de precio PVPC."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from src import database as db
from src.features import FEATURES, build_features


def _gbm(**kw):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=30, l2_regularization=1.0,
                                         random_state=42, **kw)


def _fit_models(X: pd.DataFrame, y: pd.Series) -> dict:
    # Mediana (punto) + cuantiles 10 y 90 para la banda de incertidumbre
    return {
        "pred": _gbm(loss="absolute_error").fit(X, y),
        "p10": _gbm(loss="quantile", quantile=0.10).fit(X, y),
        "p90": _gbm(loss="quantile", quantile=0.90).fit(X, y),
    }


def train() -> dict:
    prices = db.read_prices()
    data = build_features(prices).dropna(subset=FEATURES + ["y"])
    cutoff = data.index.max() - pd.Timedelta(days=config.VALIDATION_DAYS)
    train_df, val_df = data[data.index <= cutoff], data[data.index > cutoff]

    # 1) Validación temporal (sin barajar): entrenar con el pasado, medir en las 2 últimas semanas
    models = _fit_models(train_df[FEATURES], train_df["y"])
    pred = models["pred"].predict(val_df[FEATURES])
    baseline = val_df["lag_168"].values           # ingenuo estacional: "igual que hace una semana"
    y = val_df["y"].values
    mae = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mape = float(np.mean(np.abs((y - pred) / np.clip(np.abs(y), 1, None))) * 100)
    base_mae = mean_absolute_error(y, baseline)
    # Calibración conformal (CQR): un modelo entrenado hasta 30 días antes del corte
    # predice esos 30 días; sus errores fuera de la banda dicen cuánto ensancharla.
    cal_cut = cutoff - pd.Timedelta(days=30)
    fit_df, cal_df = train_df[train_df.index <= cal_cut], train_df[train_df.index > cal_cut]
    cal_models = {k: _gbm(loss="quantile", quantile=q).fit(fit_df[FEATURES], fit_df["y"])
                  for k, q in (("p10", 0.10), ("p90", 0.90))}
    yc = cal_df["y"].values
    scores = np.maximum(cal_models["p10"].predict(cal_df[FEATURES]) - yc,
                        yc - cal_models["p90"].predict(cal_df[FEATURES]))
    n = len(scores)
    cqr_offset = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.80) / n)))

    lo, hi = models["p10"].predict(val_df[FEATURES]), models["p90"].predict(val_df[FEATURES])
    raw_coverage = float(np.mean((y >= lo) & (y <= hi)) * 100)
    coverage = float(np.mean((y >= lo - cqr_offset) & (y <= hi + cqr_offset)) * 100)

    imp = permutation_importance(models["pred"], val_df[FEATURES], y, n_repeats=5,
                                 random_state=0, scoring="neg_mean_absolute_error")
    importances = dict(sorted(zip(FEATURES, imp.importances_mean.round(3)),
                              key=lambda kv: -kv[1]))

    db.save_backtest(pd.DataFrame({"real": y, "pred": pred, "baseline": baseline},
                                  index=val_df.index))

    # 2) Modelo final con todos los datos
    final = _fit_models(data[FEATURES], data["y"])
    final["cqr_offset"] = cqr_offset
    joblib.dump(final, config.MODEL_PATH)

    run = dict(trained_at=pd.Timestamp.now(tz="UTC").isoformat(), n_train=len(data),
               mae=round(mae, 2), rmse=round(rmse, 2), mape=round(mape, 2),
               baseline_mae=round(base_mae, 2),
               importances=json.dumps({**{k: float(v) for k, v in importances.items()},
                                       "_coverage": round(coverage, 1),
                                       "_raw_coverage": round(raw_coverage, 1)}))
    db.save_run(**run)
    return run


def forecast() -> pd.DataFrame:
    models = joblib.load(config.MODEL_PATH)
    offset = models.pop("cqr_offset", 0.0)
    prices = db.read_prices()
    feats = build_features(prices, extend_hours=config.FORECAST_HORIZON_H)
    future = feats[feats["y"].isna() & (feats.index > prices.index.max())]
    future = future.iloc[:config.FORECAST_HORIZON_H]
    X = future[FEATURES]
    out = pd.DataFrame({k: m.predict(X) for k, m in models.items()}, index=X.index)
    out["p10"] = np.minimum(out["p10"] - offset, out["pred"])
    out["p90"] = np.maximum(out["p90"] + offset, out["pred"])
    db.save_forecast(out.round(2), pd.Timestamp.now(tz="UTC"))
    return out
