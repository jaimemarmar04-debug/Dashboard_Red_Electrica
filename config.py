"""Configuración central del proyecto."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "pvpc.db"
MODEL_PATH = BASE_DIR / "data" / "model.joblib"

# API de Red Eléctrica (REData). Pública, sin clave.
REE_URL = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
REE_CHUNK_DAYS = 20          # la API limita el rango por petición en granularidad horaria
REE_TIMEOUT = 20

# Si la API no responde (sin red, mantenimiento...) se usa un generador sintético
# con la misma forma de datos para poder seguir desarrollando y demostrando.
# Solo rellena horas que falten: nunca sobrescribe datos reales de REE.
# Se desactiva con la variable de entorno ALLOW_SYNTHETIC_FALLBACK=0.
ALLOW_SYNTHETIC_FALLBACK = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "1") == "1"

HISTORY_DAYS = 400           # histórico que se descarga la primera vez
FORECAST_HORIZON_H = 48      # horas que predice el modelo
VALIDATION_DAYS = 14         # holdout temporal para métricas
TIMEZONE = "Europe/Madrid"

UPDATE_EVERY_MIN = 15        # frecuencia de consulta a la API
RETRAIN_EVERY_H = 24         # frecuencia de reentrenamiento
UI_REFRESH_SECONDS = 60      # refresco del dashboard

APPLIANCES = {
    "Lavadora": {"hours": 2, "kw": 2.0},
    "Lavavajillas": {"hours": 2, "kw": 1.5},
    "Secadora": {"hours": 2, "kw": 2.5},
    "Horno": {"hours": 1, "kw": 2.2},
    "Coche eléctrico": {"hours": 5, "kw": 7.4},
    "Termo eléctrico": {"hours": 3, "kw": 1.5},
}

# Servidor (Render inyecta PORT)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8050"))
