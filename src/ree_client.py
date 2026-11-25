"""Cliente de la API REData de Red Eléctrica."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

import config


def _parse_response(payload: dict) -> pd.DataFrame:
    frames = []
    for item in payload.get("included", []):
        title = item.get("attributes", {}).get("title", "").lower()
        if "pvpc" in title:
            col = "pvpc"
        elif "spot" in title or "mercado" in title:
            col = "spot"
        else:
            continue
        values = item["attributes"].get("values", [])
        if not values:
            continue
        df = pd.DataFrame(values)[["datetime", "value"]].rename(columns={"value": col})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        frames.append(df.set_index("datetime"))
    if not frames:
        return pd.DataFrame(columns=["pvpc", "spot"])
    df = pd.concat(frames, axis=1)
    # Desde 2025 el mercado es cuartohorario: se agrega a hora
    return df.resample("1h").mean()


def fetch_prices(start: datetime, end: datetime) -> pd.DataFrame:
    """Descarga precios horarios (UTC) entre start y end, troceando el rango."""
    out = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=config.REE_CHUNK_DAYS), end)
        params = {
            "start_date": cursor.strftime("%Y-%m-%dT%H:%M"),
            "end_date": chunk_end.strftime("%Y-%m-%dT%H:%M"),
            "time_trunc": "hour",
        }
        r = requests.get(config.REE_URL, params=params, timeout=config.REE_TIMEOUT,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        out.append(_parse_response(r.json()))
        cursor = chunk_end
    df = pd.concat(out)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(how="all")
