# Cuándo enchufar

Dashboard que consulta de forma continua el precio horario de la luz (PVPC) en la API pública
de Red Eléctrica, lo guarda en una base de datos SQLite, entrena un modelo de machine learning
que predice las próximas 48 horas y ayuda a decidir a qué hora poner cada electrodoméstico.

Asignatura: Desarrollo de Aplicaciones para la Visualización de Datos (ICAI).
Autor: Jaime Martínez.

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                         # http://127.0.0.1:8050
```

La primera ejecución descarga unos 400 días de histórico y entrena el modelo (menos de un minuto).
A partir de ahí, mientras la app esté abierta, cada 15 minutos se consulta la API, se guardan los
datos nuevos, se reentrena el modelo si tiene más de 24 h y se recalcula la previsión. El dashboard
se refresca solo cada minuto.

Para ejecutar el ciclo de datos sin interfaz (por ejemplo, desde cron): `python -m src.pipeline`.

## Modo simulado

Si la API no responde (sin conexión, mantenimiento, bloqueo de red), la app usa un generador sintético
con la misma estructura y rasgos del mercado español y lo indica en la cabecera con un punto ámbar.
Se desactiva poniendo `ALLOW_SYNTHETIC_FALLBACK = False` en `config.py`. Para empezar de cero con
datos reales, borra la carpeta `data/`.

## Estructura

```
config.py            parámetros (API, horizonte, frecuencias, electrodomésticos)
app.py               dashboard Dash + planificador de tareas (APScheduler)
src/ree_client.py    cliente de la API REData (troceo por rangos, agregado a horas)
src/synthetic.py     generador de datos de respaldo
src/database.py      SQLite: prices, forecasts, backtest, model_runs, ingest_log
src/features.py      variables de calendario, retardos y medias móviles
src/model.py         entrenamiento, validación, calibración e inferencia
src/pipeline.py      ingesta, reentrenamiento y predicción
assets/style.css     estilos
```

## Modelo

- Objetivo: PVPC (€/MWh) de cada una de las próximas 48 horas.
- Variables: hora, día de la semana, mes, festivo, codificación cíclica, precios retrasados 48, 72,
  96, 168 y 336 h, medias y volatilidad móviles retrasadas y precio mayorista retrasado. Al usar solo
  retardos de 48 h o más, un único modelo directo predice todo el horizonte sin fuga de información.
- Algoritmo: `HistGradientBoostingRegressor` (mediana) más dos modelos cuantílicos (p10 y p90).
- Incertidumbre: la banda p10 a p90 se calibra con conformal quantile regression sobre una ventana
  de 30 días separada, para que cubra realmente el 80 % de los casos.
- Validación: holdout temporal de los últimos 14 días frente a la referencia "igual que hace una
  semana". Métricas: MAE, RMSE, MAPE, mejora sobre la referencia y cobertura de la banda.
- Explicabilidad: importancia por permutación en el conjunto de validación.
