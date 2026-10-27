"""EDA reproducible sobre las fuentes reales: figuras en eda/figures/, reportes en eda/reports/.

Cubre: congestión crónica por corredor y hora (SIMM 2017-2020), los 6,49 M de viajes
diarios de la EOD 2025 y su reparto modal y horario, la densidad origen-destino por
macrozona, la composición vehicular (aforos con ~40 % de motos), la siniestralidad
2008-2025 (276 muertes en 2025 según el Observatorio), el efecto de la lluvia y el
cambio de régimen de la pandemia.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402
from .fuentes import cargar_eod_json  # noqa: E402

log = logging.getLogger(__name__)

AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICO = "#e34948"
TINTA, TINTA_2, GRILLA, FONDO, NEUTRO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b2"
SECUENCIAL = LinearSegmentedColormap.from_list("mobi_seq", ["#eef4fc", "#9cc3ee", AZUL, "#0d3f7a"])
FUENTE_SENSORES = "Fuente: MEData — Velocidad y tiempo de viaje GT (SIMM, Alcaldía de Medellín)"
FUENTE_EOD = "Fuente: EOD 2025, visor oficial del AMVA (eod2025.metropol.gov.co)"


def _estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRILLA,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "figure.dpi": 110,
    })


def _fuente(fig, texto: str) -> None:
    fig.text(0.01, 0.005, texto, fontsize=7.5, color=TINTA_2, ha="left", va="bottom")


def _guardar(fig, nombre: str) -> str:
    ruta = config.EDA_FIGURES / nombre
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(ruta.relative_to(config.ROOT))


def _miles(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _datos() -> dict:
    sens = pd.read_parquet(config.SENSORES_STD)
    tramos = pd.read_csv(config.TRAMOS_STD)
    pre = sens[sens["timestamp"] < pd.Timestamp(config.FIN_TEST)]
    vref = pre.groupby("tramo_id")["velocidad_kmh"].quantile(config.PCT_VEL_LIBRE).rename("vref")
    sens = sens.join(vref, on="tramo_id")
    sens["indice_congestion"] = (1 - sens["velocidad_kmh"] / sens["vref"]).clip(lower=0)
    sens["colapso"] = sens["velocidad_kmh"] / sens["vref"] < config.COLLAPSE_SPEED_RATIO
    nombre = tramos.set_index("tramo_id").apply(lambda r: f"{r['corredor']} ({r['sentido']})", axis=1)
    sens["tramo"] = sens["tramo_id"].map(nombre)
    return {
        "sens": sens, "tramos": tramos,
        "clima": pd.read_parquet(config.CLIMA_STD),
        "eod": cargar_eod_json(),
        "eod_hora": pd.read_csv(config.EOD_HORA_STD),
        "od": pd.read_csv(config.EOD_OD_STD),
        "mz": pd.read_csv(config.MACROZONAS_FILE),
        "siniestros": pd.read_parquet(config.SINIESTROS_STD) if config.SINIESTROS_STD.exists() else None,
        "aforos": pd.read_parquet(config.AFOROS_STD) if config.AFOROS_STD.exists() else None,
        "festivos": set(pd.read_csv(config.FESTIVOS_STD, parse_dates=["fecha"])["fecha"].dt.normalize()),
    }


def _laborable_pre(s: pd.DataFrame, festivos: set) -> pd.DataFrame:
    return s[(s["timestamp"] < pd.Timestamp(config.FIN_TEST)) & (s["timestamp"].dt.day_of_week < 5)
             & ~s["timestamp"].dt.normalize().isin(festivos)]


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_congestion(D) -> tuple[str, pd.DataFrame]:
    s = _laborable_pre(D["sens"], D["festivos"])
    piv = s.pivot_table(index="tramo", columns=s["timestamp"].dt.hour, values="indice_congestion", aggfunc="mean")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    im = ax.imshow(piv.values, aspect="auto", cmap=SECUENCIAL, vmin=0, vmax=0.6)
    ax.set_xticks(range(piv.shape[1]), [f"{h:02d}h" for h in piv.columns])
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8.5)
    ax.grid(False)
    perfil_h = piv.mean(axis=0)
    h_man = int(perfil_h[perfil_h.index < 12].idxmax()) if (perfil_h.index < 12).any() else None
    h_tar = int(perfil_h[perfil_h.index >= 12].idxmax()) if (perfil_h.index >= 12).any() else None
    picos = " y ".join(f"{h} h" for h in [h_man, h_tar] if h is not None)
    ax.set_title(f"Congestión crónica: picos a las {picos} (más oscuro = más lento que lo normal)")
    ax.set_xlabel("Hora (días laborables, feb-2017 a mar-2020)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Índice de congestión = 1 − vel / vel de referencia (p95)", color=TINTA_2)
    cb.outline.set_visible(False)
    _fuente(fig, FUENTE_SENSORES)
    return _guardar(fig, "01_congestion_cronica_heatmap.png"), piv


def fig_eod_horas(D) -> str:
    h = D["eod_hora"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    series = [("viajes_publico", "Transporte público", AZUL), ("viajes_no_motorizado", "No motorizado", NARANJA),
              ("viajes_privado", "Privado (auto, moto)", AQUA), ("viajes_informal", "Informal", AMARILLO)]
    for col, nom, c in series:
        ax.plot(h["hora"], h[col] / 1e3, color=c, lw=2, marker="o", ms=4, label=nom)
    ax.set_xticks(range(0, 24, 2))
    total = D["eod"]["indicadores"]["1"]["valor"]
    ax.set_title(f"{_miles(total)} viajes en un día hábil (EOD 2025): pico de salida a las 6 h; a pie, también a las 12 h")
    ax.set_xlabel("Hora de inicio del viaje")
    ax.set_ylabel("Miles de viajes")
    ax.legend(loc="upper right", ncols=2)
    _fuente(fig, FUENTE_EOD)
    return _guardar(fig, "02_eod2025_viajes_por_hora.png")


def fig_eod_modos(D) -> tuple[str, dict]:
    ind = D["eod"]["indicadores"]
    m = pd.DataFrame(ind["27"]["grupos"]).set_index("criterio")["valor"].sort_values() * 100
    col = {"Motocicleta": NARANJA, "Masivo": AZUL, "TPC": AZUL}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(m.index, m.values, color=[col.get(k, NEUTRO) for k in m.index], height=0.68)
    for y, v in enumerate(m.values):
        ax.text(v + 0.4, y, f"{v:.1f} %", va="center", fontsize=9, color=TINTA)
    moto, auto = ind["13"]["valor"], ind["12"]["valor"]
    ax.set_title("Reparto modal EOD 2025: a pie lidera y la moto ya supera al automóvil")
    ax.set_xlabel("% de viajes (modo principal)")
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.06, f"Motorización: {moto:.1f} motos y {auto:.1f} autos por 1.000 habitantes",
            transform=ax.transAxes, ha="right", fontsize=9, color=TINTA_2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NARANJA), plt.Rectangle((0, 0), 1, 1, color=AZUL),
                       plt.Rectangle((0, 0), 1, 1, color=NEUTRO)],
              labels=["Moto", "Transporte público (masivo y TPC)", "Otros"], loc="lower right",
              bbox_to_anchor=(1, 0.12))
    _fuente(fig, FUENTE_EOD + ". TPC = transporte público colectivo")
    return _guardar(fig, "03_eod2025_reparto_modal.png"), m.round(2).to_dict()


def fig_densidad_od(D) -> tuple[str, pd.DataFrame]:
    od, mz, tr = D["od"], D["mz"].dropna(subset=["lat"]), D["tramos"]
    atr = od.groupby("destino")["viajes"].sum().rename("atraidos")
    gen = od.groupby("origen")["viajes"].sum().rename("generados")
    z = mz.join(atr, on="id_macrozona").join(gen, on="id_macrozona").fillna(0)
    z = z[z["lon"] > -75.70]   # Barbosa y Girardota quedan fuera del recorte del valle central
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    sc = ax.scatter(z["lon"], z["lat"], s=z["atraidos"] / 900, c=z["atraidos"] / 1e3, cmap=SECUENCIAL,
                    edgecolors=FONDO, linewidths=1.5, alpha=0.95, vmin=0)
    top = z.nlargest(6, "atraidos").sort_values("lat")
    for k, (_, r) in enumerate(top.iterrows()):
        dx = 14 if k % 2 == 0 else -14
        ax.annotate(f"{r['macrozona']}\n{_miles(r['atraidos'])}", (r["lon"], r["lat"]), xytext=(dx, 0),
                    textcoords="offset points", fontsize=8, color=TINTA, va="center",
                    ha="left" if dx > 0 else "right")
    ax.scatter(tr["lon"], tr["lat"], marker="s", s=28, color=NARANJA, label="Corredores con sensor (SIMM)", zorder=3)
    ax.set_xlim(-75.70, -75.48)
    ax.set_ylim(6.03, 6.38)
    ax.set_aspect("equal")
    ax.set_title("Densidad de destinos: el centro y El Poblado atraen más viajes")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Miles de viajes atraídos por día hábil", color=TINTA_2)
    cb.outline.set_visible(False)
    ax.legend(loc="lower right")
    _fuente(fig, FUENTE_EOD + " (matriz OD por macrozona); ubicación de cámaras: MEData/SIMM")
    return _guardar(fig, "04_eod2025_densidad_destinos.png"), z.sort_values("atraidos", ascending=False)


def fig_siniestralidad(D) -> tuple[str | None, dict]:
    s = D["siniestros"]
    if s is None:
        return None, {}
    s = s[(s["anio"] >= 2008) & (s["anio"] <= 2025)]
    anual = s.groupby("anio").agg(incidentes=("fatal", "size"), fatales=("fatal", "sum"))
    hora = s.dropna(subset=["hora"]).groupby("hora")["fatal"].agg(["size", "sum"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    colores = [NEUTRO if a == 2025 else AZUL for a in anual.index]
    ax.bar(anual.index, anual["fatales"], color=colores, width=0.75)
    ax.set_title("Incidentes con al menos un muerto, por año")
    ref = config.REFERENCIAS["muertes_viales_medellin_2025"]
    ax.text(0.02, 0.97, f"2025 (gris): solo ene-ago en la base.\nCierre oficial 2025: {ref['valor']} personas "
            f"fallecidas\n({ref['detalle']}).", transform=ax.transAxes, va="top", fontsize=8.5, color=TINTA_2)
    ax.set_ylim(0, anual["fatales"].max() * 1.35)
    ax.set_xticks(anual.index[::2])
    ax2 = axes[1]
    ax2.bar(hora.index, hora["size"] / 1e3, color=AZUL, width=0.8)
    ax2.set_title("Incidentes por hora del día (2008-2025)")
    ax2.set_xlabel("Hora")
    ax2.set_ylabel("Miles de incidentes")
    ax2.set_xticks(range(0, 24, 3))
    fig.suptitle("Siniestralidad vial en Medellín", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TINTA)
    _fuente(fig, "Fuente: Correa Álvarez et al. (2026), Mendeley Data doi:10.17632/r6g5dfnpgh.1 (Secretaría de "
                 "Movilidad); cierre 2025: Observatorio de Movilidad vía El Colombiano")
    return _guardar(fig, "05_siniestralidad_vial.png"), {
        "incidentes_2008_2025": int(len(s)), "incidentes_fatales": int(s["fatal"].sum()),
        "fatales_por_anio": {int(k): int(v) for k, v in anual["fatales"].items()},
        "hora_con_mas_incidentes": int(hora["size"].idxmax())}


def fig_colapsos(D) -> tuple[str, pd.Series]:
    s = _laborable_pre(D["sens"], D["festivos"])
    tasa = (s.groupby("tramo")["colapso"].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(tasa.index, tasa.values, color=AZUL, height=0.7)
    for y, v in enumerate(tasa.values):
        ax.text(v + 0.2, y, f"{v:.1f} %", va="center", fontsize=8.5, color=TINTA)
    ax.set_title(f"Horas en colapso por tramo (vel < {config.COLLAPSE_SPEED_RATIO:.0%} de la de referencia)")
    ax.set_xlabel("% de horas laborables 6-20 h, pre-pandemia")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", visible=False)
    _fuente(fig, FUENTE_SENSORES)
    return _guardar(fig, "06_horas_en_colapso_por_tramo.png"), tasa


def fig_pandemia(D) -> str:
    s = D["sens"][(D["sens"]["timestamp"] >= "2019-01-01")]
    s = s[s["timestamp"].dt.hour.isin([7, 8, 17, 18])]
    dia = s.groupby(s["timestamp"].dt.normalize())["indice_congestion"].mean()
    sem = dia.rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(sem.index, sem.values, color=AZUL, lw=1.8)
    ax.axvline(pd.Timestamp(config.FIN_TEST), color=CRITICO, lw=1.2, ls="--")
    ax.annotate("20-mar-2020: inicia el aislamiento\n(comienza la ventana de drift)",
                (pd.Timestamp(config.FIN_TEST), sem.max() * 0.95), xytext=(8, 0), textcoords="offset points",
                fontsize=8.5, color=TINTA)
    ax.set_title("La pandemia cambió el régimen: la congestión en hora pico se desplomó")
    ax.set_ylabel("Índice de congestión en pico (media 7 días)")
    ax.set_ylim(0, None)
    _fuente(fig, FUENTE_SENSORES)
    return _guardar(fig, "07_drift_pandemia.png")


def fig_lluvia(D) -> tuple[str, dict]:
    s = _laborable_pre(D["sens"], D["festivos"]).merge(D["clima"][["timestamp", "lluvia_mm"]], on="timestamp",
                                                         how="left")
    s = s[s["timestamp"].dt.hour.isin([7, 8, 17, 18, 19])]
    cortes = pd.cut(s["lluvia_mm"].fillna(0), [-0.01, 0.0, 1, 3, 1000], labels=["Sin lluvia", "0-1 mm", "1-3 mm", "> 3 mm"])
    r = s.groupby(cortes, observed=True).agg(indice=("indice_congestion", "mean"), colapso=("colapso", "mean"),
                                             horas=("colapso", "size"))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(r.index.astype(str), r["colapso"] * 100, color=[NEUTRO, "#9cc3ee", AZUL, "#0d3f7a"], width=0.6)
    for i, (v, n) in enumerate(zip(r["colapso"] * 100, r["horas"])):
        ax.text(i, v + 0.3, f"{v:.1f} %\n(n={_miles(n)})", ha="center", fontsize=8.5, color=TINTA)
    ax.set_title("Con lluvia fuerte, el colapso en hora pico es más frecuente")
    ax.set_ylabel("% de horas pico en colapso")
    ax.set_ylim(0, r["colapso"].max() * 130)
    ax.grid(axis="x", visible=False)
    _fuente(fig, FUENTE_SENSORES + "; lluvia: Open-Meteo (ERA5)")
    return _guardar(fig, "08_lluvia_vs_colapso.png"), (r["colapso"] * 100).round(2).to_dict()


def fig_aforos(D) -> tuple[str | None, dict]:
    a = D["aforos"]
    if a is None:
        return None, {}
    g = a.groupby("hora")[["autos", "buses", "camiones", "motos", "bicicletas"]].sum()
    pct = g.div(g.sum(axis=1), axis=0) * 100
    pct = pct[(pct.index >= 5) & (pct.index <= 21)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(pct.index, pct["motos"], color=NARANJA, lw=2.2, marker="o", ms=4, label="Motos")
    ax.plot(pct.index, pct["autos"], color=AZUL, lw=2.2, marker="o", ms=4, label="Autos")
    ax.plot(pct.index, pct["buses"] + pct["camiones"], color=NEUTRO, lw=2, label="Buses y camiones")
    tot = a[["autos", "buses", "camiones", "motos", "bicicletas"]].sum()
    ax.set_title(f"Composición vehicular aforada: {tot['motos'] / tot.sum() * 100:.0f} % son motos")
    ax.set_xlabel("Hora")
    ax.set_ylabel("% de vehículos aforados")
    ax.set_ylim(0, 65)
    ax.legend(loc="upper right")
    _fuente(fig, f"Fuente: MEData — Aforos vehiculares {int(a['anio'].min())}-{int(a['anio'].max())}, "
                 f"{int(a['nodos'].max())} intersecciones (Secretaría de Movilidad)")
    return _guardar(fig, "09_composicion_vehicular_aforos.png"), {
        "pct_motos": round(float(tot["motos"] / tot.sum() * 100), 1),
        "pct_autos": round(float(tot["autos"] / tot.sum() * 100), 1)}


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def ejecutar_eda() -> dict:
    config.ensure_dirs()
    _estilo()
    D = _datos()
    s, ind = D["sens"], D["eod"]["indicadores"]

    figuras = {}
    figuras["congestion"], piv = fig_congestion(D)
    figuras["eod_horas"] = fig_eod_horas(D)
    figuras["eod_modos"], modos = fig_eod_modos(D)
    figuras["densidad_od"], zonas = fig_densidad_od(D)
    f_sin, sin = fig_siniestralidad(D)
    if f_sin:
        figuras["siniestralidad"] = f_sin
    figuras["colapsos"], tasa = fig_colapsos(D)
    figuras["pandemia"] = fig_pandemia(D)
    figuras["lluvia"], lluvia = fig_lluvia(D)
    f_af, afo = fig_aforos(D)
    if f_af:
        figuras["aforos"] = f_af

    # Calidad de la serie de sensores
    dias = s["timestamp"].dt.normalize().nunique()
    cobertura = (s.groupby("tramo_id").size() / (dias * 15) * 100).round(1)
    calidad = pd.DataFrame({"lecturas": s.groupby("tramo_id").size(), "cobertura_pct": cobertura,
                            "vel_media_kmh": s.groupby("tramo_id")["velocidad_kmh"].mean().round(1),
                            "vel_referencia_p95": s.groupby("tramo_id")["vref"].first().round(1)})
    calidad.to_csv(config.EDA_REPORTS / "calidad_sensores_por_tramo.csv")
    zonas[["id_macrozona", "municipio", "macrozona", "atraidos", "generados"]].round(0).to_csv(
        config.EDA_REPORTS / "eod2025_viajes_por_macrozona.csv", index=False)

    resumen = {
        "sensores": {"lecturas": int(len(s)), "tramos": int(s["tramo_id"].nunique()), "dias": int(dias),
                     "periodo": [str(s["timestamp"].min()), str(s["timestamp"].max())],
                     "cobertura_media_pct": float(cobertura.mean().round(1)),
                     "tramo_mas_congestionado": str(piv.mean(axis=1).idxmax()),
                     "hora_pico": int(piv.mean(axis=0).idxmax()),
                     "pct_horas_colapso_pre_pandemia": round(float(tasa.mean()), 2),
                     "tramo_mas_colapsos": str(tasa.idxmax())},
        "eod2025": {"viajes_dia_habil": round(ind["1"]["valor"]), "pct_no_viaja": round(ind["2"]["valor"] * 100, 2),
                    "tiempo_promedio_min": round(ind["3"]["valor"], 1),
                    "motos_por_1000_hab": round(ind["13"]["valor"], 1),
                    "autos_por_1000_hab": round(ind["12"]["valor"], 1), "reparto_modal_pct": modos,
                    "macrozona_mas_atractora": str(zonas.iloc[0]["municipio"] + " - " + zonas.iloc[0]["macrozona"]),
                    "viajes_por_municipio_residencia": D["eod"].get("viajes_por_municipio_residencia", {})},
        "siniestralidad": sin, "aforos": afo, "lluvia_pct_colapso_pico": lluvia,
        "referencias_publicadas": config.REFERENCIAS, "figuras": figuras,
    }
    (config.EDA_REPORTS / "eda_summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str),
                                                         encoding="utf-8")
    _reporte(resumen, tasa, calidad)
    log.info("EDA completo: %d figuras", len(figuras))
    return resumen


def _reporte(r: dict, tasa: pd.Series, calidad: pd.DataFrame) -> None:
    se, eo, si = r["sensores"], r["eod2025"], r["siniestralidad"]
    ref = config.REFERENCIAS
    L = [
        "# Reporte EDA — MobiPredict (datos reales)",
        "",
        "Generado por `src/mobipredict/eda.py`. Figuras en `eda/figures/`. Fuentes en `docs/fuentes_de_datos.md`.",
        "",
        "## Cifras clave",
        "",
        "| Indicador | Valor | Fuente |",
        "| --- | --- | --- |",
        f"| Viajes en un día hábil | {_miles(eo['viajes_dia_habil'])} | EOD 2025 (API AMVA) |",
        f"| Personas que no viajan | {eo['pct_no_viaja']} % | EOD 2025 |",
        f"| Tiempo promedio de viaje | {eo['tiempo_promedio_min']} min | EOD 2025 |",
        f"| Motos / autos por 1.000 habitantes | {eo['motos_por_1000_hab']} / {eo['autos_por_1000_hab']} | EOD 2025 |",
        f"| Variación tenencia de motos 2017-2025 | +{ref['variacion_tenencia_motos_2017_2025_pct']['valor']:.0f} % | "
        "EOD 2025, publicado por El Colombiano |",
        f"| Variación participación TP 2017-2025 | {ref['variacion_participacion_tp_2017_2025_pct']['valor']:.0f} % | "
        "EOD 2025, publicado por El Colombiano |",
        f"| Muertes viales Medellín 2025 | {ref['muertes_viales_medellin_2025']['valor']} | Observatorio de Movilidad |",
        f"| Lecturas de velocidad (tramo-hora) | {_miles(se['lecturas'])} en {se['tramos']} tramos, {se['dias']} días | "
        "MEData / SIMM |",
        f"| Horas en colapso (laborable, pre-pandemia) | {se['pct_horas_colapso_pre_pandemia']} % | MEData / SIMM |",
    ]
    if si:
        L.append(f"| Incidentes viales 2008-ago 2025 | {_miles(si['incidentes_2008_2025'])} "
                 f"({_miles(si['incidentes_fatales'])} fatales) | Mendeley Data |")
    if r.get("aforos"):
        L.append(f"| Motos en los aforos | {r['aforos']['pct_motos']} % de los vehículos | MEData aforos |")
    L += [
        "",
        "## Hallazgos",
        "",
        f"1. **Congestión crónica.** Se repite a diario en las horas pico; {se['tramo_mas_congestionado']} es el tramo "
        "más congestionado. El perfil histórico por tramo y hora es una línea base fuerte y una feature clave.",
        f"2. **Demanda.** La EOD 2025 registra {_miles(eo['viajes_dia_habil'])} viajes en un día hábil, con pico de "
        "inicio a las 6 h. La distribución horaria oficial entra al modelo como prior de demanda.",
        f"3. **Densidad.** {eo['macrozona_mas_atractora']} es la macrozona que más viajes atrae. Cada tramo hereda los "
        "viajes generados y atraídos de su macrozona como feature espacial.",
        f"4. **Motos.** Hay {eo['motos_por_1000_hab']} motos por cada 1.000 habitantes, casi el doble que autos. "
        + (f"En los aforos son el {r['aforos']['pct_motos']} % de los vehículos." if r.get("aforos") else ""),
        "5. **Lluvia.** Con más de 3 mm/h el colapso en hora pico es más frecuente; la lluvia actual y la de las "
        "últimas 3 horas entran como features.",
        "6. **Pandemia.** Desde el 20-mar-2020 la congestión se desplomó. Ese periodo se aparta del entrenamiento y "
        "se usa como prueba de estrés de drift.",
    ]
    if si:
        L.append(f"7. **Siniestralidad.** {_miles(si['incidentes_2008_2025'])} incidentes reportados entre 2008 y agosto "
                 f"de 2025; la hora con más incidentes es las {si['hora_con_mas_incidentes']} h. El cierre oficial de "
                 f"2025 fue de {ref['muertes_viales_medellin_2025']['valor']} muertes "
                 f"({ref['muertes_viales_medellin_2025']['detalle']}).")
    L += [
        "",
        "## Horas en colapso por tramo",
        "",
        "| Tramo | % horas |",
        "| --- | --- |",
        *[f"| {k} | {v:.2f} |" for k, v in tasa.sort_values(ascending=False).items()],
        "",
        "## Calidad de la serie de sensores",
        "",
        "Cobertura = lecturas / (días con datos × 15 horas). Detalle en `calidad_sensores_por_tramo.csv`.",
        "",
        f"Cobertura media: {se['cobertura_media_pct']} %. Las lecturas solo existen de 6:00 a 20:00; los rezagos "
        "no cruzan la noche y las horas faltantes no se imputan.",
        "",
        "## Limitaciones",
        "",
        "- La serie de velocidad abierta llega a diciembre de 2020; la EOD es de 2025. Se asume que la forma horaria "
        "de la demanda y la jerarquía de macrozonas son estables; antes de operar hay que reentrenar con el feed "
        "actual del SIMM.",
        "- Las velocidades provienen de tiempos de viaje de Google Traffic agregados por la Alcaldía, no de espiras.",
        "- La base de siniestros cuenta incidentes (un incidente puede tener varias víctimas); el cierre oficial cuenta "
        "personas fallecidas.",
    ]
    (config.EDA_REPORTS / "eda_report.md").write_text("\n".join(L), encoding="utf-8")
