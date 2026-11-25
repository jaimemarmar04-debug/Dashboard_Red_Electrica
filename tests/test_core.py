import numpy as np
import pandas as pd

from src import database as db
from src import synthetic
from src.features import FEATURES, LAGS, build_features


def _prices(days=30):
    end = pd.Timestamp("2026-03-01", tz="UTC")
    return synthetic.generate(end - pd.Timedelta(days=days), end)


def test_synthetic_es_determinista_y_horario():
    a, b = _prices(), _prices()
    pd.testing.assert_frame_equal(a, b)
    assert (a.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert {"pvpc", "spot"} <= set(a.columns)


def test_features_sin_fuga_de_informacion():
    """Cambiar el precio de la última hora no puede alterar ninguna variable de las
    47 horas siguientes: todos los retardos de precio son de 48 h o más."""
    p = _prices()
    X1 = build_features(p, extend_hours=48)
    p2 = p.copy()
    p2.iloc[-1, p2.columns.get_loc("pvpc")] += 500
    X2 = build_features(p2, extend_hours=48)
    futuro = X1.index > p.index[-1]
    cols = [c for c in FEATURES if c != "spot_lag_48"]
    pd.testing.assert_frame_equal(X1.loc[futuro, cols].iloc[:47], X2.loc[futuro, cols].iloc[:47])
    assert min(LAGS) >= 48


def test_features_horizonte_completo():
    p = _prices()
    X = build_features(p, extend_hours=48)
    futuro = X[X.index > p.index[-1]]
    assert len(futuro) == 48
    assert futuro["y"].isna().all()
    assert futuro[FEATURES].notna().all().all()


def test_mejor_ventana():
    from app import best_window
    idx = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    outlook = pd.DataFrame({"price": [300, 100, 50, 60, 400, 200]}, index=idx)
    bw = best_window(outlook, hours=2, kw=1.0)
    assert bw["best_start"] == idx[2]                 # 50 + 60
    assert np.isclose(bw["best"], 0.11)               # €/kWh * kW * h
    assert bw["worst_start"] == idx[4] or bw["worst_start"] == idx[3]
    assert best_window(outlook, hours=10, kw=1.0) is None


def test_simulado_no_pisa_datos_reales(tmp_db):
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    real = pd.DataFrame({"pvpc": [100.0, 110.0, 120.0], "spot": [50.0, 55.0, 60.0]}, index=idx)
    db.upsert_prices(real, "REE")
    fake = pd.DataFrame({"pvpc": [1.0, 1.0, 1.0, 1.0], "spot": [1.0] * 4},
                        index=pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"))
    db.upsert_prices(fake, "simulado")
    out = db.read_prices()
    assert out["pvpc"].iloc[:3].tolist() == [100.0, 110.0, 120.0]
    assert out["source"].tolist() == ["REE", "REE", "REE", "simulado"]
    # y los datos reales sí sustituyen a los simulados
    db.upsert_prices(fake.iloc[3:].assign(pvpc=130.0), "REE")
    assert db.read_prices()["source"].iloc[3] == "REE"
    assert db.last_timestamp(source="REE") == idx[-1] + pd.Timedelta(hours=1)
