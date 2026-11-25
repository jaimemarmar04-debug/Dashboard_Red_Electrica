"""Persistencia en SQLite: precios, predicciones y ejecuciones del modelo."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    datetime TEXT PRIMARY KEY,   -- UTC ISO
    pvpc REAL, spot REAL, source TEXT, ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
    created_at TEXT, datetime TEXT, pred REAL, p10 REAL, p90 REAL,
    PRIMARY KEY (created_at, datetime)
);
CREATE TABLE IF NOT EXISTS backtest (
    datetime TEXT PRIMARY KEY, real REAL, pred REAL, baseline REAL
);
CREATE TABLE IF NOT EXISTS model_runs (
    trained_at TEXT PRIMARY KEY, n_train INTEGER, mae REAL, rmse REAL,
    mape REAL, baseline_mae REAL, importances TEXT
);
CREATE TABLE IF NOT EXISTS ingest_log (
    run_at TEXT PRIMARY KEY, source TEXT, rows INTEGER, status TEXT
);
"""


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


def _to_df(con, query, params=()):
    df = pd.read_sql(query, con, params=params)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.set_index("datetime")


def upsert_prices(df: pd.DataFrame, source: str) -> int:
    """Guarda precios. Los datos reales sobrescriben cualquier fila (incluidas las
    simuladas); los simulados solo rellenan horas vacías y nunca pisan datos reales."""
    if df.empty:
        return 0
    now = pd.Timestamp.now(tz="UTC").isoformat()
    rows = [(ts.isoformat(), _f(r.get("pvpc")), _f(r.get("spot")), source, now)
            for ts, r in df.iterrows()]
    if source == "simulado":
        sql = "INSERT INTO prices VALUES (?,?,?,?,?) ON CONFLICT(datetime) DO NOTHING"
    else:
        sql = ("INSERT INTO prices VALUES (?,?,?,?,?) ON CONFLICT(datetime) DO UPDATE SET "
               "pvpc=excluded.pvpc, spot=excluded.spot, source=excluded.source, "
               "ingested_at=excluded.ingested_at")
    with connect() as con:
        cur = con.executemany(sql, rows)
        return cur.rowcount if cur.rowcount >= 0 else len(rows)


def read_prices(since: pd.Timestamp | None = None) -> pd.DataFrame:
    q, params = "SELECT datetime, pvpc, spot, source FROM prices", ()
    if since is not None:
        q += " WHERE datetime >= ?"
        params = (since.isoformat(),)
    with connect() as con:
        return _to_df(con, q + " ORDER BY datetime", params)


def last_timestamp(source: str | None = None) -> pd.Timestamp | None:
    q, params = "SELECT MAX(datetime) FROM prices WHERE pvpc IS NOT NULL", ()
    if source is not None:
        q += " AND source = ?"
        params = (source,)
    with connect() as con:
        r = con.execute(q, params).fetchone()[0]
    return pd.Timestamp(r) if r else None


def save_forecast(df: pd.DataFrame, created_at: pd.Timestamp):
    rows = [(created_at.isoformat(), ts.isoformat(), r.pred, r.p10, r.p90)
            for ts, r in df.iterrows()]
    with connect() as con:
        con.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?)", rows)


def latest_forecast() -> pd.DataFrame:
    with connect() as con:
        return _to_df(con, "SELECT datetime, pred, p10, p90 FROM forecasts WHERE created_at = "
                           "(SELECT MAX(created_at) FROM forecasts) ORDER BY datetime")


def save_backtest(df: pd.DataFrame):
    with connect() as con:
        con.execute("DELETE FROM backtest")
        con.executemany("INSERT INTO backtest VALUES (?,?,?,?)",
                        [(ts.isoformat(), r.real, r.pred, r.baseline) for ts, r in df.iterrows()])


def read_backtest() -> pd.DataFrame:
    with connect() as con:
        return _to_df(con, "SELECT * FROM backtest ORDER BY datetime")


def save_run(**kw):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO model_runs VALUES (?,?,?,?,?,?,?)",
                    (kw["trained_at"], kw["n_train"], kw["mae"], kw["rmse"],
                     kw["mape"], kw["baseline_mae"], kw["importances"]))


def latest_run() -> dict | None:
    with connect() as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM model_runs ORDER BY trained_at DESC LIMIT 1").fetchone()
    return dict(r) if r else None


def log_ingest(source: str, rows: int, status: str):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?)",
                    (pd.Timestamp.now(tz="UTC").isoformat(), source, rows, status))


def last_ingest() -> dict | None:
    with connect() as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM ingest_log ORDER BY run_at DESC LIMIT 1").fetchone()
    return dict(r) if r else None
