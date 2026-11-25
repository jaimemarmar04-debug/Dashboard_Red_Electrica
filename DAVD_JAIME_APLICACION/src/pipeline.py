"""Orquestación: ingesta periódica desde la API, reentrenamiento y predicción."""
from __future__ import annotations

import logging

import pandas as pd

import config
from src import database as db
from src import model, ree_client, synthetic

log = logging.getLogger("pipeline")


def update_data() -> str:
    """Descarga lo que falte desde el último dato (con solape de 2 días para
    recoger correcciones) y lo guarda. Devuelve la fuente usada."""
    now = pd.Timestamp.now(tz="UTC")
    last = db.last_timestamp()
    start = (last - pd.Timedelta(days=2)) if last is not None else now - pd.Timedelta(days=config.HISTORY_DAYS)
    end = now.normalize() + pd.Timedelta(days=2)  # REE publica el día siguiente por la tarde
    try:
        df = ree_client.fetch_prices(start.to_pydatetime(), end.to_pydatetime())
        if df.empty:
            raise ValueError("respuesta vacía")
        source = "REE"
    except Exception as exc:  # noqa: BLE001
        if not config.ALLOW_SYNTHETIC_FALLBACK:
            db.log_ingest("REE", 0, f"error: {exc}")
            raise
        log.warning("API REE no disponible (%s). Usando datos simulados.", exc)
        # En modo simulado solo existen datos hasta la hora actual (como en tiempo real)
        df = synthetic.generate(start, now.floor("h"))
        source = "simulado"
    n = db.upsert_prices(df, source)
    db.log_ingest(source, n, "ok")
    log.info("Ingesta %s: %d filas", source, n)
    return source


def retrain_if_needed(force: bool = False):
    run = db.latest_run()
    stale = run is None or (pd.Timestamp.now(tz="UTC") - pd.Timestamp(run["trained_at"])
                            > pd.Timedelta(hours=config.RETRAIN_EVERY_H))
    if force or stale or not config.MODEL_PATH.exists():
        log.info("Reentrenando modelo...")
        r = model.train()
        log.info("MAE %.2f vs baseline %.2f", r["mae"], r["baseline_mae"])


def run_cycle(force_retrain: bool = False):
    update_data()
    retrain_if_needed(force_retrain)
    model.forecast()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cycle(force_retrain=True)
