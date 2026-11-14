"""Monitor de TIDIS - Streamlit page.

Every visit (at most once per hour) downloads the file of outstanding securities,
saves the TIDIS "image" of the day, rebuilds the daily history (imputing missing
days), makes/saves the forecast of the day and flags significant changes.
Run the same cycle without the page with:  python collect.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from tidis import pipeline
from tidis.alerts import SEV_ORDER, forecast_accuracy
from tidis.analysis import compare_images, month_profile
from tidis.config import SETTINGS, SOURCE_URL
from tidis.forecast import estimate_model
from tidis.storage import Store

st.set_page_config(page_title="Monitor TIDIS", page_icon="📉", layout="wide")

UNIT = 1e9
UNIT_LABEL = "miles de millones COP"
store = Store()


def cop(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def mm(x: float) -> str:
    """Thousands of millions of COP; sign first so st.metric colours deltas correctly."""
    return f"{'-' if x < 0 else ''}${abs(x) / UNIT:,.1f} mil M"


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Descargando el archivo de títulos vigentes…")
def auto_fetch(hour_key: str) -> dict:
    """Automatic download, at most once per hour per server process."""
    return pipeline.fetch_and_ingest(store)


@st.cache_data(show_spinner="Reconstruyendo la historia de los TIDIS…")
def load(version: float) -> pipeline.History:
    h = pipeline.load_history(store)
    if not h.empty:
        h.alerts = pipeline.refresh_alerts(store, h)
    return h


def data_version() -> float:
    # alerts.csv is excluded on purpose: it is an output of load()
    return sum(p.stat().st_mtime_ns for p in (store.catalog_path, store.vintages_path) if p.exists())


def show_fetch_result(res: dict | None, where=st.sidebar) -> None:
    if not res:
        return
    if res["status"] == "error":
        where.warning(res["error"] + "\n\nSe muestra la historia guardada.")
    elif res["status"] == "unchanged":
        where.info(f"Imagen del {res['date']} ya estaba guardada (sin cambios).")
    else:
        msg = "nueva" if res["status"] == "new" else "actualizada"
        extra = f" · {res['imputed_rows']} filas imputadas" if res.get("imputed_rows") else ""
        where.success(f"Imagen del {res['date']} {msg}: {res['rows']} títulos, {cop(res['total'])}{extra}")


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("📉 Monitor TIDIS")
if st.sidebar.button("Descargar ahora", type="primary", width="stretch"):
    auto_fetch.clear()
    with st.spinner("Descargando…"):
        st.session_state["manual_fetch"] = pipeline.fetch_and_ingest(store)
fetch_result = st.session_state.pop("manual_fetch", None) or auto_fetch(pipeline.now_local().strftime("%Y-%m-%d %H"))
show_fetch_result(fetch_result)

with st.sidebar.expander("Agregar archivo histórico"):
    st.caption("Sube un archivo 'características financieras títulos vigentes' de una fecha pasada "
               "para completar la historia.")
    up = st.file_uploader("Archivo .xlsx", type=["xlsx"])
    up_date = st.date_input("Fecha de la imagen (vacío = leerla del archivo)", value=None,
                            max_value=pipeline.today())
    if up is not None and st.button("Guardar imagen"):
        try:
            res = pipeline.ingest(store, up.getvalue(), day=up_date, origin="carga manual")
            show_fetch_result(res)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")

n_new = pipeline.refresh_forecasts(store)   # forecast vintage for every new image (write-once)
h = load(data_version())

st.sidebar.divider()
st.sidebar.caption(f"Fuente: [archivo de títulos vigentes]({SOURCE_URL})  \n"
                   f"Historia guardada en `{store.root}`  \n"
                   f"{len(h.catalog)} imágenes observadas · {int(h.imputed.sum())} días imputados")

# --------------------------------------------------------------------------- header
st.title("TIDIS: evolución, predicciones y alertas")
if h.empty:
    st.info("Todavía no hay imágenes guardadas. Descarga el archivo (barra lateral) o sube uno histórico.")
    st.stop()

last_day = h.panel.index[-1]
image = h.panel.loc[last_day]
age = (pd.Timestamp(pipeline.today()) - last_day).days
if age > SETTINGS.stale_days:
    st.warning(f"La última imagen es del {last_day.date()} ({age} días). Las cifras de hoy no están observadas; "
               "ver la pestaña *Predicciones* para la estimación.")

obs_days = list(h.obs.index)
prev_obs = obs_days[-2] if len(obs_days) > 1 else None
total = float(image.sum())
last30 = h.flows[h.flows.index > last_day - pd.Timedelta(days=30)]
latest_v = h.vintages.iloc[-1] if not h.vintages.empty else None

c = st.columns(5)
c[0].metric("Saldo TIDIS en circulación", mm(total),
            delta=mm(total - h.obs.loc[prev_obs].sum()) if prev_obs is not None else None,
            help=f"{cop(total)} al {last_day.date()}" + (f" · cambio vs {prev_obs.date()}" if prev_obs is not None else ""))
c[1].metric("Usado (redimido) 30 días", mm(last30["redeemed"].sum()), help="Títulos usados antes de su vencimiento")
c[2].metric("Emitido 30 días", mm(last30["issued"].sum()))
c[3].metric("Vencido sin usar 30 días", mm(last30["matured"].sum()))
if latest_v is not None:
    c[4].metric("Saldo que vencerá sin usarse (pred.)", mm(latest_v["expected_unused_at_maturity"]),
                help="Predicción del saldo actual que llegará a su vencimiento sin ser usado")

recent = h.alerts[(h.alerts["date"] >= last_day - pd.Timedelta(days=6)) & (h.alerts["severity"] != "info")]
recent = recent.assign(o=recent["severity"].map(SEV_ORDER)).sort_values(["o", "date"], ascending=[True, False])
for _, a in recent.head(4).iterrows():
    (st.error if a["severity"] == "alerta" else st.warning)(f"**{a['date'].date()} · {a['category']}** — {a['message']}")
if len(recent) > 4:
    st.caption(f"… y {len(recent) - 4} alertas más en la pestaña *Alertas*.")

tabs = st.tabs(["Imagen actual", "Historia", "Comparar imágenes", "Predicciones", "Alertas", "Datos"])

# --------------------------------------------------------------------------- tab: current image
with tabs[0]:
    st.subheader(f"Saldo por mes de vencimiento al {last_day.date()}")
    ref_opt = {"Imagen anterior": 1, "Hace 7 días": 7, "Hace 30 días": 30, "Hace 90 días": 90}
    ref_label = st.segmented_control("Comparar con", list(ref_opt), default="Imagen anterior")
    ref_day = None
    if ref_label == "Imagen anterior":
        ref_day = prev_obs
    elif ref_label:
        cand = last_day - pd.Timedelta(days=ref_opt[ref_label])
        ref_day = cand if cand >= h.panel.index[0] else None
    prof = month_profile(image).rename("Actual")
    df = prof.to_frame()
    if ref_day is not None:
        df = df.join(month_profile(h.panel.loc[ref_day]).rename(f"{ref_day.date()}"), how="outer").fillna(0)
    long = (df / UNIT).reset_index(names="mes").melt("mes", var_name="imagen", value_name="saldo")
    long["mes"] = long["mes"].astype(str)
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            x=alt.X("mes:N", title="Mes de vencimiento"), xOffset="imagen:N",
            y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})"), color=alt.Color("imagen:N", title="Imagen"),
            tooltip=["mes", "imagen", alt.Tooltip("saldo:Q", format=",.1f")]),
        width="stretch")
    if ref_day is not None:
        st.caption(f"Imagen de referencia: {ref_day.date()}" + (" (imputada)" if h.imputed.get(ref_day, False) else ""))
    if not h.fc_buckets.empty and latest_v is not None:
        b = h.fc_buckets[h.fc_buckets["asof"] == latest_v["asof"]].copy()
        b["mes"] = b["maturity_date"].dt.to_period("M").astype(str)
        bm = b.groupby("mes")[["expected_used", "expected_at_maturity"]].sum().div(UNIT)
        bm.columns = ["Se usará (pred.)", "Vencerá sin usarse (pred.)"]
        st.markdown("**Destino esperado del saldo actual** (según la predicción del día)")
        bl = bm.reset_index().melt("mes", var_name="destino", value_name="saldo")
        st.altair_chart(alt.Chart(bl).mark_bar().encode(
            x=alt.X("mes:N", title="Mes de vencimiento"), y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})"),
            color=alt.Color("destino:N", title=None, scale=alt.Scale(range=["#4c78a8", "#e45756"])),
            tooltip=["mes", "destino", alt.Tooltip("saldo:Q", format=",.1f")]), width="stretch")
    with st.expander("Tabla por fecha de vencimiento"):
        t = image[image > 0].rename("Saldo en circulación").to_frame()
        t.index = t.index.date
        t.index.name = "Fecha de vencimiento"
        t["Días al vencimiento"] = [(pd.Timestamp(d) - last_day).days for d in t.index]
        st.dataframe(t, column_config={"Saldo en circulación": st.column_config.NumberColumn(format="dollar")})

# --------------------------------------------------------------------------- tab: history
with tabs[1]:
    st.subheader("Saldo total en circulación")
    fl = h.flows.reset_index()
    fl["tipo"] = fl["imputed"].map({True: "imputado", False: "observado"})
    fl["saldo"] = fl["total"] / UNIT
    line = alt.Chart(fl).mark_line(color="#4c78a8").encode(
        x=alt.X("date:T", title=None), y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})", scale=alt.Scale(zero=False)))
    pts = alt.Chart(fl).mark_circle(size=28).encode(
        x="date:T", y="saldo:Q",
        color=alt.Color("tipo:N", title=None, scale=alt.Scale(domain=["observado", "imputado"], range=["#4c78a8", "#f58518"])),
        tooltip=[alt.Tooltip("date:T", title="fecha"), "tipo", alt.Tooltip("saldo:Q", format=",.1f")])
    marks = []
    flagged = h.alerts[h.alerts["severity"] == "alerta"].drop_duplicates("date")
    if not flagged.empty:
        marks.append(alt.Chart(flagged).mark_rule(color="#e45756", strokeDash=[3, 3]).encode(
            x="date:T", tooltip=[alt.Tooltip("date:T", title="alerta"), "message"]))
    st.altair_chart(alt.layer(line, pts, *marks), width="stretch")
    st.caption("Líneas rojas: días con alertas. Puntos naranja: días sin imagen, imputados.")

    st.subheader("¿Por qué cambia el saldo?")
    freq = st.segmented_control("Agrupar por", ["Día", "Semana", "Mes"], default="Semana", key="freq")
    rule = {"Día": "D", "Semana": "W", "Mes": "ME"}[freq or "Semana"]
    if len(h.flows) < 2:
        st.info("Se necesitan al menos dos imágenes para descomponer los cambios.")
    else:
        comp = h.flows[["issued", "redeemed", "matured"]].iloc[1:].resample(rule).sum() / UNIT
        comp["redeemed"] *= -1
        comp["matured"] *= -1
        comp.columns = ["Emitido", "Usado (redimido)", "Vencido sin usar"]
        cl = comp.reset_index().melt("date", var_name="componente", value_name="monto")
        st.altair_chart(alt.Chart(cl).mark_bar().encode(
            x=alt.X("date:T", title=None), y=alt.Y("monto:Q", title=f"Cambio ({UNIT_LABEL})"),
            color=alt.Color("componente:N", title=None,
                            scale=alt.Scale(domain=list(comp.columns), range=["#54a24b", "#4c78a8", "#e45756"])),
            tooltip=[alt.Tooltip("date:T", title="periodo"), "componente", alt.Tooltip("monto:Q", format=",.1f")]),
            width="stretch")

    st.subheader("Evolución de la imagen (saldo por mes de vencimiento)")
    by_month = h.panel.T.groupby(h.panel.columns.to_period("M")).sum().T
    step = max(1, len(by_month) // 200)
    hm = (by_month.iloc[::step] / UNIT).reset_index(names="date").melt("date", var_name="mes", value_name="saldo")
    hm["mes"] = hm["mes"].astype(str)
    hm = hm[hm["saldo"] > 0]
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x=alt.X("date:T", title="Fecha de la imagen"), y=alt.Y("mes:O", title="Mes de vencimiento", sort="descending"),
        color=alt.Color("saldo:Q", title=UNIT_LABEL, scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("date:T", title="imagen"), "mes", alt.Tooltip("saldo:Q", format=",.1f")]),
        width="stretch")

    st.subheader("Degradación por cohorte")
    st.caption("Cada línea es un mes de vencimiento: % de su saldo máximo observado que sigue en circulación "
               "(desde que alcanzó ese máximo), según los días que le faltan para vencer.")
    rows = []
    for m in by_month.columns[-14:]:
        s = by_month[m]
        s = s[s > 0]
        if len(s) and len(s.loc[s.idxmax():]) >= 2:
            s = s.loc[s.idxmax():]              # from full issuance onwards: pure degradation
        if len(s) < 2:
            continue
        end = m.to_timestamp(how="end").normalize()
        rows.append(pd.DataFrame({"dias_al_vencimiento": (end - s.index).days,
                                  "pct": 100 * s.values / s.max(), "cohorte": str(m)}))
    if rows:
        coh = pd.concat(rows)
        st.altair_chart(alt.Chart(coh).mark_line().encode(
            x=alt.X("dias_al_vencimiento:Q", title="Días al vencimiento", scale=alt.Scale(reverse=True)),
            y=alt.Y("pct:Q", title="% del saldo máximo"), color=alt.Color("cohorte:N", title="Vence en"),
            tooltip=["cohorte", "dias_al_vencimiento", alt.Tooltip("pct:Q", format=".1f")]), width="stretch")

# --------------------------------------------------------------------------- tab: compare
with tabs[2]:
    st.subheader("Comparar dos imágenes")
    first, last = h.panel.index[0].date(), last_day.date()
    cc = st.columns(2)
    d_a = cc[0].date_input("Imagen A", value=(prev_obs or last_day).date(), min_value=first, max_value=last)
    d_b = cc[1].date_input("Imagen B", value=last, min_value=first, max_value=last)
    d_a, d_b = pd.Timestamp(d_a), pd.Timestamp(d_b)
    for lbl, d in (("A", d_a), ("B", d_b)):
        if h.imputed.get(d, False):
            st.caption(f"La imagen {lbl} ({d.date()}) no fue observada: es imputada.")
    if d_a >= d_b:
        st.info("Elige una imagen A anterior a la imagen B.")
    else:
        cmp_df = compare_images(h.panel.loc[d_a], h.panel.loc[d_b], d_a, d_b, h.alive_on_maturity)
        tot = cmp_df[["balance_a", "balance_b", "issued", "redeemed", "matured"]].sum()
        k = st.columns(5)
        k[0].metric("Saldo A", mm(tot["balance_a"]))
        k[1].metric("Saldo B", mm(tot["balance_b"]), delta=mm(tot["balance_b"] - tot["balance_a"]))
        k[2].metric("Emitido", mm(tot["issued"]))
        k[3].metric("Usado", mm(tot["redeemed"]))
        k[4].metric("Vencido sin usar", mm(tot["matured"]))
        view = cmp_df.copy()
        view.index = view.index.astype(str)
        cl = (view[["balance_a", "balance_b"]] / UNIT).rename(columns={"balance_a": f"A {d_a.date()}",
                                                                      "balance_b": f"B {d_b.date()}"})
        cl = cl.reset_index(names="mes").melt("mes", var_name="imagen", value_name="saldo")
        st.altair_chart(alt.Chart(cl).mark_bar().encode(
            x=alt.X("mes:N", title="Mes de vencimiento"), xOffset="imagen:N",
            y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})"), color=alt.Color("imagen:N", title=None),
            tooltip=["mes", "imagen", alt.Tooltip("saldo:Q", format=",.1f")]), width="stretch")
        view.columns = ["Saldo A", "Saldo B", "Emitido", "Usado", "Vencido sin usar", "Cambio", "Cambio %"]
        view[view.columns[:-1]] = view[view.columns[:-1]] / UNIT
        view.index.name = "Mes de vencimiento"
        st.caption(f"Cifras en {UNIT_LABEL}")
        money = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(view, column_config={c: money for c in view.columns[:-1]} |
                     {"Cambio %": st.column_config.NumberColumn(format="%.1f%%")})

# --------------------------------------------------------------------------- tab: forecasts
with tabs[3]:
    if latest_v is None:
        st.info("Aún no hay predicciones.")
    else:
        asof = latest_v["asof"]
        st.subheader(f"Predicción del saldo total (hecha con datos al {asof.date()})")
        if latest_v["model"] == "runoff":
            st.info("Con una sola imagen no se puede estimar el ritmo de uso: la predicción solo descuenta los "
                    "vencimientos (escenario sin uso ni nuevas emisiones). Mejora a medida que se guardan imágenes diarias.")
        horizon = st.slider("Horizonte (días)", 30, SETTINGS.horizon_days, 180, step=30)
        fc = h.fc_totals[(h.fc_totals["asof"] == asof)]
        fc = fc[fc["target_date"] <= asof + pd.Timedelta(days=horizon)]
        hist = h.flows.reset_index()[["date", "total"]]
        hist = hist[hist["date"] >= asof - pd.Timedelta(days=horizon)]
        area = fc.melt("target_date", ["existing", "new_issues"], var_name="c", value_name="v")
        area["c"] = area["c"].map({"existing": "Títulos actuales", "new_issues": "Nuevas emisiones (pred.)"})
        area["v"] /= UNIT
        ch = alt.layer(
            alt.Chart(area).mark_area(opacity=0.55).encode(
                x=alt.X("target_date:T", title=None), y=alt.Y("v:Q", stack=True, title=f"Saldo ({UNIT_LABEL})"),
                color=alt.Color("c:N", title=None, scale=alt.Scale(range=["#9ecae9", "#c7e9c0"])),
                tooltip=[alt.Tooltip("target_date:T", title="fecha"), "c", alt.Tooltip("v:Q", format=",.1f")]),
            alt.Chart(hist.assign(v=hist["total"] / UNIT)).mark_line(color="black").encode(
                x="date:T", y="v:Q", tooltip=[alt.Tooltip("date:T", title="fecha"), alt.Tooltip("v:Q", format=",.1f")]),
        )
        st.altair_chart(ch, width="stretch")
        if age > 0:
            now_pred = fc[fc["target_date"] == pd.Timestamp(pipeline.today())]
            if len(now_pred):
                st.caption(f"Estimación para hoy ({pipeline.today()}): {mm(now_pred['total'].iloc[0])}")

        m = st.columns(4)
        m[0].metric("Uso diario promedio", f"{100 * latest_v['global_daily_rate']:.3f}%",
                    help="Fracción del saldo vigente que se usa (redime) por día")
        m[1].metric("Emisión diaria promedio", mm(latest_v["issuance_per_day"]))
        m[2].metric("Vencerá sin usarse", mm(latest_v["expected_unused_at_maturity"]))
        m[3].metric("Imágenes usadas", int(latest_v["n_snapshots"]),
                    help="Modelo: " + ("uso + vencimientos + nuevas emisiones" if latest_v["model"] == "usage"
                                       else "solo vencimientos"))

        model = estimate_model(h.panel, h.alive_on_maturity, h.snapshots)
        if model.kind == "usage" and len(model.rates):
            st.markdown("**Curva de degradación estimada**: % del saldo que se usa por día según los días al vencimiento")
            rc = pd.DataFrame({"dias": model.rates.index, "tasa": 100 * model.rates.values})
            st.altair_chart(alt.Chart(rc).mark_bar().encode(
                x=alt.X("dias:O", title="Días al vencimiento (inicio del tramo)"),
                y=alt.Y("tasa:Q", title="% diario usado"),
                tooltip=["dias", alt.Tooltip("tasa:Q", format=".3f")]), width="stretch")

        st.subheader("¿Cómo han cambiado las predicciones?")
        vint = h.fc_totals.merge(h.vintages[["asof"]], on="asof")
        if vint["asof"].nunique() < 2:
            st.info("Se necesita más de una predicción guardada para comparar.")
        else:
            n_show = st.slider("Predicciones a mostrar", 2, min(60, vint["asof"].nunique()),
                               min(15, vint["asof"].nunique()))
            keep = sorted(vint["asof"].unique())[-n_show:]
            vv = vint[vint["asof"].isin(keep) & (vint["target_date"] <= last_day + pd.Timedelta(days=horizon))].copy()
            vv["v"] = vv["total"] / UNIT
            vv["predicción del"] = vv["asof"].dt.strftime("%Y-%m-%d")
            act = h.actual_obs[h.actual_obs.index >= min(keep)].rename("v").rename_axis("date").reset_index()
            act["v"] /= UNIT
            st.altair_chart(alt.layer(
                alt.Chart(vv).mark_line(strokeWidth=1.2).encode(
                    x=alt.X("target_date:T", title=None), y=alt.Y("v:Q", title=f"Saldo ({UNIT_LABEL})", scale=alt.Scale(zero=False)),
                    color=alt.Color("predicción del:O", scale=alt.Scale(scheme="viridis")),
                    tooltip=["predicción del", alt.Tooltip("target_date:T", title="fecha"), alt.Tooltip("v:Q", format=",.1f")]),
                alt.Chart(act).mark_line(color="black", strokeWidth=2.5).encode(x="date:T", y="v:Q"),
            ), width="stretch")
            st.caption("Negro: saldo observado. Colores: trayectoria prevista por cada predicción guardada.")

            rows = []
            for a, g in vint.groupby("asof"):
                g = g.set_index("target_date")["total"]
                for hz in SETTINGS.check_horizons:
                    t = a + pd.Timedelta(days=hz)
                    if t in g.index:
                        rows.append({"asof": a, "serie": f"+{hz} días", "v": g[t] / UNIT})
            vm = h.vintages.set_index("asof")["expected_unused_at_maturity"]
            rows += [{"asof": a, "serie": "Vencerá sin usarse", "v": v / UNIT} for a, v in vm.items()]
            hz_df = pd.DataFrame(rows)
            st.markdown("**Predicción a horizonte fijo según la fecha en que se hizo**: un salto = cambio significativo")
            pred_alerts = h.alerts[h.alerts["category"].isin(["predicción", "error"])].drop_duplicates("date")
            layers = [alt.Chart(hz_df).mark_line(point=True).encode(
                x=alt.X("asof:T", title="Fecha de la predicción"), y=alt.Y("v:Q", title=UNIT_LABEL, scale=alt.Scale(zero=False)),
                color=alt.Color("serie:N", title=None),
                tooltip=[alt.Tooltip("asof:T", title="predicción del"), "serie", alt.Tooltip("v:Q", format=",.1f")])]
            if not pred_alerts.empty:
                layers.append(alt.Chart(pred_alerts).mark_rule(color="#e45756", strokeDash=[3, 3]).encode(
                    x="date:T", tooltip=[alt.Tooltip("date:T", title="alerta"), "message"]))
            st.altair_chart(alt.layer(*layers), width="stretch")

            acc = forecast_accuracy(h.fc_totals, h.actual_obs)
            if not acc.empty:
                st.markdown("**Precisión histórica de las predicciones** (contra imágenes observadas)")
                summ = acc.groupby("horizon_bin", observed=True).agg(
                    predicciones=("error_pct", "size"), error_medio=("error_pct", "mean"),
                    error_absoluto_medio=("error_pct", lambda x: x.abs().mean()))
                summ.index.name = "Horizonte"
                pct = st.column_config.NumberColumn
                st.dataframe(summ, column_config={
                    "predicciones": pct("Comparaciones"),
                    "error_medio": pct("Sesgo (pred. − real)", format="%.2f%%"),
                    "error_absoluto_medio": pct("Error absoluto medio", format="%.2f%%")})
        with st.expander("Cómo funciona la predicción"):
            st.markdown(f"""
* Cada día se guarda la **imagen** del saldo de TIDIS por fecha de vencimiento.
* Comparando imágenes consecutivas se mide qué fracción de cada vencimiento se **usa** (se redime para
  pagar impuestos) por día, según los días que le faltan para vencer (tramos de {SETTINGS.dtm_bin_days} días,
  últimos {SETTINGS.lookback_days} días de historia).
* La predicción proyecta cada vencimiento con esa curva de degradación; al llegar su fecha, el saldo que
  quede **vence sin usarse**. Se agregan nuevas emisiones al ritmo promedio observado con vigencia de
  {SETTINGS.tenor_days} días.
* Cada predicción se guarda una sola vez (no se reescribe), para poder ver cómo fueron cambiando y medir su error.
* Alertas: revisión > {SETTINGS.forecast_change_pct:.0f}% entre predicciones consecutivas (horizontes
  {', '.join(str(x) for x in SETTINGS.check_horizons)} días), error > {SETTINGS.forecast_error_pct:.0f}% de la
  predicción anterior contra el dato observado, y uso/emisión inusual (z robusto > {SETTINGS.z_threshold}).
""")

# --------------------------------------------------------------------------- tab: alerts
with tabs[4]:
    st.subheader("Registro de alertas")
    if h.alerts.empty:
        st.success("Sin alertas.")
    else:
        f = st.columns(2)
        cats = f[0].multiselect("Categoría", sorted(h.alerts["category"].unique()), default=sorted(h.alerts["category"].unique()))
        sevs = f[1].multiselect("Severidad", ["alerta", "advertencia", "info"], default=["alerta", "advertencia"])
        al = h.alerts[h.alerts["category"].isin(cats) & h.alerts["severity"].isin(sevs)].sort_values("date", ascending=False)
        st.dataframe(al[["date", "severity", "category", "message", "first_detected"]], hide_index=True,
                     column_config={"date": st.column_config.DateColumn("Fecha"),
                                    "severity": st.column_config.TextColumn("Severidad", width="small"),
                                    "category": st.column_config.TextColumn("Categoría", width="small"),
                                    "message": st.column_config.TextColumn("Detalle", width="large"),
                                    "first_detected": st.column_config.TextColumn("Detectada el", width="small")})
        st.download_button("Descargar alertas (CSV)", h.alerts.to_csv(index=False).encode("utf-8"), "alertas_tidis.csv")

# --------------------------------------------------------------------------- tab: data
with tabs[5]:
    st.subheader("Imágenes guardadas")
    tc = st.column_config.TextColumn
    st.dataframe(h.catalog.sort_values("snapshot_date", ascending=False), hide_index=True,
                 column_config={"snapshot_date": st.column_config.DateColumn("Fecha imagen"),
                                "saved_at": tc("Guardada el"), "source_cut_date": tc("Fecha de corte (archivo)"),
                                "origin": tc("Origen"), "n_rows": st.column_config.NumberColumn("Títulos"),
                                "n_imputed_rows": st.column_config.NumberColumn("Filas imputadas"),
                                "total_balance": st.column_config.NumberColumn("Saldo total", format="dollar"),
                                "sha256": None})
    imp_days = h.imputed[h.imputed].index
    st.markdown(f"**Días imputados:** {len(imp_days)}" +
                (f" — {'… ' if len(imp_days) > 15 else ''}{', '.join(str(d.date()) for d in imp_days[-15:])}"
                 if len(imp_days) else ""))
    st.caption(f"Método: títulos vigentes a ambos lados del hueco → interpolación '{SETTINGS.imputation}'; "
               "títulos que vencen en el hueco → saldo constante hasta su vencimiento; "
               "títulos nuevos → aparecen en su fecha de emisión estimada. "
               "Filas con saldo o vencimiento faltante → se completan con la imagen anterior del mismo título.")
    d = st.columns(3)
    d[0].download_button("Historia diaria (CSV)", h.flows.to_csv().encode("utf-8"), "historia_tidis.csv")
    panel_long = h.panel.stack().rename("saldo").reset_index()
    panel_long.columns = ["fecha_imagen", "fecha_vencimiento", "saldo"]
    panel_long = panel_long[panel_long["saldo"] > 0].assign(imputado=lambda x: x["fecha_imagen"].map(h.imputed))
    d[1].download_button("Imágenes diarias (CSV)", panel_long.to_csv(index=False).encode("utf-8"), "imagenes_tidis.csv")
    raw = store.raw_path(h.obs.index[-1])
    if raw.exists():
        d[2].download_button("Archivo original más reciente", raw.read_bytes(), raw.name)
    if n_new:
        st.caption(f"{n_new} predicción(es) nuevas guardadas en esta visita.")
