"""Cuándo enchufar: dashboard de previsión del precio de la luz (PVPC).

Ejecutar:  python app.py   ->  http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from dash import Dash, Input, Output, State, dcc, html

import config
from src import database as db
from src import pipeline
from src.features import FEATURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

C = dict(ink="#18222E", muted="#5B6B7A", line="#CBD3DB", valle="#2F8F5B",
         llano="#D99A1E", punta="#C4473A", forecast="#2F5FA7", band="rgba(47,95,167,0.16)")
TZ = config.TIMEZONE


# ---------- utilidades ----------
def kwh(x):  # €/MWh -> €/kWh
    return x / 1000.0


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def dh(ts):  # "mié 04:00"
    return f"{DIAS[ts.dayofweek]} {ts:%H}:00"


def eur(x, d=3):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def base_layout(fig: go.Figure, height=320):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=C["ink"], size=13),
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        separators=",.")
    fig.update_xaxes(showgrid=False, linecolor=C["line"], ticks="outside", tickcolor=C["line"],
                     tickformat="%d/%m", hoverformat="%d/%m %H:00")
    fig.update_yaxes(gridcolor="#E6EBEF", zeroline=False)
    return fig


def load_state():
    now = pd.Timestamp.now(tz="UTC").floor("h")
    prices = db.read_prices(since=now - pd.Timedelta(days=95))
    fc = db.latest_forecast()
    known_future = prices.loc[prices.index >= now, ["pvpc"]].rename(columns={"pvpc": "price"})
    known_future["kind"] = "oficial"
    fc_future = fc.loc[(fc.index >= now) & (~fc.index.isin(known_future.index)), ["pred"]] \
        .rename(columns={"pred": "price"})
    fc_future["kind"] = "prevision"
    outlook = pd.concat([known_future, fc_future]).sort_index().iloc[:48]
    return now, prices, fc, outlook


def best_window(outlook: pd.DataFrame, hours: int, kw: float):
    p = kwh(outlook["price"].values)
    if len(p) < hours:
        return None
    costs = np.convolve(p, np.ones(hours), "valid") * kw
    i_best, i_worst = int(costs.argmin()), int(costs.argmax())
    return dict(best_start=outlook.index[i_best], best=costs[i_best],
                worst_start=outlook.index[i_worst], worst=costs[i_worst], now=costs[0],
                idx_best=i_best)


# ---------- layout ----------
app = Dash(__name__, title="Cuándo enchufar", external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800"
    "&family=Source+Sans+3:wght@400;600&display=swap"])
server = app.server   # servidor Flask que usa gunicorn en producción (gunicorn app:server)

app.layout = html.Div(className="wrap", children=[
    dcc.Interval(id="tick", interval=config.UI_REFRESH_SECONDS * 1000),
    html.Div(className="top", children=[
        html.Div([html.H1("Cuándo enchufar"),
                  html.P("Precio de la luz con tarifa regulada (PVPC) y previsión a 48 horas, "
                         "para mover el consumo a las horas baratas.")]),
        html.Div(id="status", className="status"),
    ]),
    html.Section(id="hero", className="hero"),
    html.Div(className="grid", children=[
        html.Div(className="panel wide", children=[
            html.H2("Evolución y previsión"),
            html.P("Precio real y previsión del modelo con su banda de confianza del 80 %.", className="sub"),
            dcc.RadioItems(id="range", className="range-radio", value=7, inline=True,
                           options=[{"label": "2 días", "value": 2}, {"label": "1 semana", "value": 7},
                                    {"label": "1 mes", "value": 30}]),
            dcc.Graph(id="g-history", config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.H2("Planifica un electrodoméstico"),
            html.P("Busca el tramo continuo más barato de las próximas 48 horas.", className="sub"),
            html.Div(className="row", children=[
                html.Div(className="field", children=["Aparato", dcc.Dropdown(
                    id="appliance", value="Lavadora", clearable=False,
                    options=list(config.APPLIANCES), style={"width": 190})]),
                html.Div(className="field", children=["Duración (h)", dcc.Input(
                    id="hours", type="number", min=1, max=12, step=1, value=2)]),
                html.Div(className="field", children=["Potencia (kW)", dcc.Input(
                    id="kw", type="number", min=0.1, max=22, step=0.1, value=2.0)]),
            ]),
            html.Div(id="plan-out"),
            dcc.Graph(id="g-plan", config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.H2("Patrón semanal"),
            html.P("Precio medio por hora y día en los últimos 90 días (€/kWh).", className="sub"),
            dcc.Graph(id="g-heat", config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.H2("¿Acierta el modelo?"),
            html.P(id="bt-sub", className="sub"),
            html.Div(id="metrics", className="metrics"),
            dcc.Graph(id="g-backtest", config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.H2("En qué se fija el modelo"),
            html.P("Cuánto empeora el error si se desordena cada variable (importancia por permutación).",
                   className="sub"),
            dcc.Graph(id="g-imp", config={"displayModeBar": False}),
        ]),
    ]),
    html.P(className="foot", children=[
        "Datos: API REData de Red Eléctrica (apidatos.ree.es). Modelo: gradient boosting con regresión "
        "cuantílica calibrada (conformal), reentrenado cada 24 h. Proyecto de la asignatura Desarrollo de Aplicaciones para la "
        "Visualización de Datos, ICAI."]),
])


# ---------- callbacks ----------
@app.callback(Output("status", "children"), Output("hero", "children"), Input("tick", "n_intervals"))
def render_hero(_):
    now, prices, _, outlook = load_state()
    ing, run = db.last_ingest(), db.latest_run()
    sim = ing and ing["source"] == "simulado"
    last_upd = pd.Timestamp(ing["run_at"]).tz_convert(TZ).strftime("%d/%m %H:%M") if ing else "nunca"
    status = [html.Div([html.Span(className="dot sim" if sim else "dot"),
                        "Datos simulados (API no disponible)" if sim else "Conectado a Red Eléctrica"]),
              html.Div(["Actualizado ", html.B(last_upd)])]
    if run:
        status.append(html.Div(["Modelo entrenado ",
                                html.B(pd.Timestamp(run["trained_at"]).tz_convert(TZ).strftime("%d/%m %H:%M"))]))

    if outlook.empty:
        return status, [html.Div("Cargando datos…")]
    current = outlook["price"].iloc[0]
    q1, q2 = np.quantile(outlook["price"], [1 / 3, 2 / 3])
    color = lambda v: C["valle"] if v <= q1 else (C["llano"] if v <= q2 else C["punta"])
    verdict = ("Buen momento para enchufar." if current <= q1 else
               "Precio intermedio: si puedes, espera." if current <= q2 else
               "Hora cara: aplaza lo que puedas.")
    bw = best_window(outlook, 2, 1.0)
    best_idx = set(range(bw["idx_best"], bw["idx_best"] + 2)) if bw else set()

    cells, ticks = [], []
    for i, (ts, r) in enumerate(outlook.iterrows()):
        local = ts.tz_convert(TZ)
        title = f"{dh(local)} ({local:%d/%m})  {eur(kwh(r.price))} €/kWh ({'oficial' if r.kind == 'oficial' else 'previsión'})"
        cls = "cell" + (" forecast" if r.kind == "prevision" else "") + (" best" if i in best_idx else "")
        cells.append(html.Div(className=cls, title=title, style={"background": color(r.price)}))
        ticks.append(html.Span(f"{local.hour}h" if local.hour % 6 == 0 else ""))

    best_txt = ""
    if bw:
        b = bw["best_start"].tz_convert(TZ)
        best_txt = f"Mejores 2 horas seguidas: desde las {b:%H}:00 del {b:%d/%m}."

    hero = [
        html.Div([html.Div("Precio ahora", className="label"),
                  html.Div([eur(kwh(current)), html.Small("€/kWh")], className="big"),
                  html.Div(verdict, className="verdict"),
                  html.Div(best_txt, className="label", style={"marginTop": 8})]),
        html.Div([html.Div("Próximas 48 horas", className="label"),
                  html.Div(cells, className="strip"), html.Div(ticks, className="ticks"),
                  html.Div(className="legend", children=[
                      html.Span([html.Span(className="sw", style={"background": C["valle"]}), "Barato"]),
                      html.Span([html.Span(className="sw", style={"background": C["llano"]}), "Medio"]),
                      html.Span([html.Span(className="sw", style={"background": C["punta"]}), "Caro"]),
                      html.Span("Rayado: previsión del modelo. Liso: precio oficial publicado."),
                  ])]),
    ]
    return status, hero


@app.callback(Output("g-history", "figure"), Input("tick", "n_intervals"), Input("range", "value"))
def render_history(_, days):
    now, prices, fc, _ = load_state()
    days = days or 7
    hist = prices.loc[prices.index >= now - pd.Timedelta(days=days)]
    fc = fc.loc[fc.index > hist.index.max()] if not hist.empty else fc
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.index.tz_convert(TZ), y=kwh(fc["p90"]), line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fc.index.tz_convert(TZ), y=kwh(fc["p10"]), fill="tonexty",
                             fillcolor=C["band"], line=dict(width=0), name="Intervalo 80 %",
                             hovertemplate="%{y:.3f}"))
    fig.add_trace(go.Scatter(x=hist.index.tz_convert(TZ), y=kwh(hist["pvpc"]), name="Precio real",
                             line=dict(color=C["ink"], width=2), hovertemplate="%{y:.3f} €/kWh"))
    fig.add_trace(go.Scatter(x=fc.index.tz_convert(TZ), y=kwh(fc["pred"]), name="Previsión",
                             line=dict(color=C["forecast"], width=2, dash="dot"),
                             hovertemplate="%{y:.3f} €/kWh"))
    fig.add_vline(x=now.tz_convert(TZ), line=dict(color=C["muted"], width=1, dash="dash"))
    fig.update_yaxes(title_text="€/kWh", tickformat=".2f")
    return base_layout(fig, 340)


@app.callback(Output("hours", "value"), Output("kw", "value"), Input("appliance", "value"))
def appliance_defaults(name):
    a = config.APPLIANCES[name]
    return a["hours"], a["kw"]


@app.callback(Output("plan-out", "children"), Output("g-plan", "figure"),
              Input("hours", "value"), Input("kw", "value"), Input("tick", "n_intervals"))
def render_plan(hours, kw, _):
    hours = min(max(int(hours or 1), 1), 12)
    kw = min(max(float(kw or 1), 0.1), 22.0)
    _, _, _, outlook = load_state()
    bw = best_window(outlook, hours, kw)
    fig = go.Figure()
    if not bw:
        return html.P("No hay suficientes horas de previsión."), base_layout(fig, 200)
    b, w = bw["best_start"].tz_convert(TZ), bw["worst_start"].tz_convert(TZ)
    saving = bw["now"] - bw["best"]
    out = html.Div(className="plan-out", children=[
        html.Div([html.Div(f"Mejor: {dh(b)}", className="k"), html.Div(f"{eur(bw['best'], 2)} €", className="v")],
                 className="best"),
        html.Div([html.Div("Si lo pones ahora", className="k"), html.Div(f"{eur(bw['now'], 2)} €", className="v")]),
        html.Div([html.Div(f"Peor: {dh(w)}", className="k"), html.Div(f"{eur(bw['worst'], 2)} €", className="v")],
                 className="worst"),
    ])
    note = html.P(f"Esperando ahorras {eur(saving, 2)} € por uso "
                  f"({eur(saving * 365, 0)} € al año si es diario).", className="sub", style={"marginTop": 10})
    x = outlook.index.tz_convert(TZ)
    colors = [C["valle"] if bw["idx_best"] <= i < bw["idx_best"] + hours else "#B8C3CD"
              for i in range(len(outlook))]
    fig.add_trace(go.Bar(x=x, y=kwh(outlook["price"]), marker_color=colors,
                         hovertemplate="%{x|%d/%m %H}:00  %{y:.3f} €/kWh<extra></extra>"))
    fig = base_layout(fig, 200)
    fig.update_xaxes(tickformat="%Hh\n%d/%m")
    fig.update_yaxes(tickformat=".2f")
    return [out, note], fig


@app.callback(Output("g-heat", "figure"), Input("tick", "n_intervals"))
def render_heat(_):
    _, prices, _, _ = load_state()
    if prices.empty:
        return base_layout(go.Figure(), 300)
    local = prices.tz_convert(TZ)
    pv = local.assign(h=local.index.hour, d=local.index.dayofweek) \
        .pivot_table(index="d", columns="h", values="pvpc", aggfunc="mean")
    days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig = go.Figure(go.Heatmap(z=kwh(pv.values), x=[f"{h}h" for h in pv.columns],
                               y=[days[i] for i in pv.index],
                               colorscale=[[0, C["valle"]], [0.5, C["llano"]], [1, C["punta"]]],
                               hovertemplate="%{y} %{x}: %{z:.3f} €/kWh<extra></extra>",
                               colorbar=dict(thickness=10, tickformat=".2f")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return base_layout(fig, 300)


@app.callback(Output("g-backtest", "figure"), Output("metrics", "children"), Output("bt-sub", "children"),
              Output("g-imp", "figure"), Input("tick", "n_intervals"))
def render_model(_):
    bt, run = db.read_backtest(), db.latest_run()
    fig = go.Figure()
    if run is None or bt.empty:
        return base_layout(fig), [], "Sin modelo todavía.", base_layout(go.Figure())
    x = bt.index.tz_convert(TZ)
    fig.add_trace(go.Scatter(x=x, y=kwh(bt["real"]), name="Real", line=dict(color=C["ink"], width=1.6)))
    fig.add_trace(go.Scatter(x=x, y=kwh(bt["pred"]), name="Modelo", line=dict(color=C["forecast"], width=1.6)))
    fig.add_trace(go.Scatter(x=x, y=kwh(bt["baseline"]), name="Igual que hace 1 semana",
                             line=dict(color=C["muted"], width=1, dash="dot"), visible="legendonly"))
    fig.update_yaxes(tickformat=".2f")
    imp = json.loads(run["importances"])
    coverage = imp.pop("_coverage", None)
    imp.pop("_raw_coverage", None)
    gain = (1 - run["mae"] / run["baseline_mae"]) * 100
    metrics = [
        html.Div([html.Div(f"{eur(kwh(run['mae']), 4)}", className="v"), html.Div("Error medio (€/kWh)", className="k")]),
        html.Div([html.Div(f"{eur(run['mape'], 1)} %", className="v"), html.Div("Error relativo", className="k")]),
        html.Div([html.Div(f"{eur(gain, 0)} %", className="v"), html.Div("Mejor que la referencia", className="k")]),
        html.Div([html.Div(f"{eur(coverage or 0, 0)} %", className="v"), html.Div("Aciertos en la banda", className="k")]),
    ]
    sub = (f"Validación en las últimas {config.VALIDATION_DAYS} jornadas, que el modelo no vio al entrenar. "
           f"Referencia: repetir el precio de hace una semana.")
    top = list(imp.items())[:10][::-1]
    fi = go.Figure(go.Bar(x=[kwh(v) for _, v in top], y=[FEATURE_LABELS.get(k, k) for k, _ in top],
                          orientation="h", marker_color=C["forecast"],
                          hovertemplate="%{y}: +%{x:.4f} €/kWh de error<extra></extra>"))
    fi.update_layout(hovermode="closest")
    fi.update_xaxes(tickformat=".3f", title_text="Aumento del error (€/kWh)")
    return base_layout(fig, 250), metrics, sub, base_layout(fi, 330)


# ---------- arranque ----------
_started = False


def start_background():
    """Arranca una sola vez por proceso: un primer ciclo en segundo plano (la web
    responde enseguida y muestra "Cargando datos…") y el planificador periódico.
    Se ejecuta al importar el módulo para que funcione igual con gunicorn."""
    global _started
    if _started or os.getenv("DISABLE_SCHEDULER") == "1":
        return
    _started = True
    threading.Thread(target=pipeline.run_cycle,
                     kwargs={"force_retrain": db.latest_run() is None}, daemon=True).start()
    sched = BackgroundScheduler(timezone=TZ)
    sched.add_job(pipeline.run_cycle, "interval", minutes=config.UPDATE_EVERY_MIN,
                  id="cycle", max_instances=1, coalesce=True)
    sched.start()


start_background()

if __name__ == "__main__":
    app.run(debug=False, host=config.HOST, port=config.PORT)
