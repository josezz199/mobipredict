"""Ejecuta SOLO la limpieza de datos y muestra qué se corrigió, paso a paso.

Uso (desde la raíz del proyecto, con los datos ya descargados en data/raw/):
    python scripts/explorar_limpieza.py

Parte 1: diagnostica los problemas en el archivo crudo de velocidad (sin modificarlo).
Parte 2: ejecuta la estandarización de todas las fuentes (fuentes.estandarizar_todo).
Parte 3: muestra las tablas limpias que quedaron en data/interim/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mobipredict import config, fuentes  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def titulo(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# Parte 1 — Diagnóstico del archivo crudo
# --------------------------------------------------------------------------- #
crudo_path = config.DATA_RAW / config.FUENTES["velocidad_gt"]["archivo"]
if not crudo_path.exists():
    sys.exit(f"No existe {crudo_path}. Primero ejecute: python scripts/descargar_datos.py")

titulo("PARTE 1 — Problemas en el archivo crudo de velocidad (MEData)")
raw = pd.read_csv(crudo_path, encoding="utf-8", encoding_errors="replace", low_memory=False)
raw.columns = fuentes.COLS_GT
print(f"Filas crudas: {len(raw):,}   Columnas: {raw.shape[1]}")
print("\nAsí se ven las primeras filas tal como vienen:")
print(raw[["fecha_hora", "dia", "mes", "anio", "hora", "corredor", "sentido", "velocidad_kmh", "diasem"]].head())

fh = pd.to_datetime(raw["fecha_hora"].astype(str).str.strip(), format="%d/%m/%Y %H:%M", errors="coerce")

# Problema 1: DÍA/MES no coinciden con FECHA_HORA
dia = pd.to_numeric(raw["dia"], errors="coerce")
mes = pd.to_numeric(raw["mes"], errors="coerce")
malas = (dia != fh.dt.day) | (mes != fh.dt.month)
print(f"\n[1] Filas donde las columnas DÍA/MES NO coinciden con FECHA_HORA: {malas.sum():,} "
      f"({malas.mean():.1%})")
if malas.any():
    print("    Ejemplos:")
    print(raw.loc[malas, ["fecha_hora", "dia", "mes", "anio"]].head(5).to_string())
coinciden = (fh.dt.day_of_week + 1 == pd.to_numeric(raw["diasem"], errors="coerce")).mean()
print(f"    En cambio, FECHA_HORA coincide con el día de la semana (DIASEM) en el {coinciden:.2%} "
      "de las filas -> se usa FECHA_HORA.")

# Problema 2: lecturas cada 30 minutos
media_hora = (fh.dt.minute == 30).sum()
print(f"\n[2] Lecturas a las :30 (resolución de 30 min): {media_hora:,}")
if media_hora:
    por_anio = fh[fh.dt.minute == 30].dt.year.value_counts().sort_index()
    print("    Por año:", por_anio.to_dict(), "-> se promedian por hora.")

# Problema 3: duplicados
clave = raw["corredor"].str.strip() + "|" + raw["sentido"].str.strip() + "|" + raw["fecha_hora"].astype(str)
dup = clave.duplicated(keep=False)
print(f"\n[3] Registros con la misma (corredor, sentido, fecha-hora): {dup.sum():,} filas "
      f"en {clave[dup].nunique():,} claves -> se promedian.")

# Problema 4: velocidades fuera de rango
v = pd.to_numeric(raw["velocidad_kmh"], errors="coerce")
fuera = ((v < 2) | (v > 110)).sum()
print(f"\n[4] Velocidades fuera de 2-110 km/h: {fuera:,} -> se descartan.")

# Problema 5: corredores con longitud o extremos variables
var = raw.groupby([raw["corredor"].str.strip(), raw["sentido"].str.strip()])["longitud_km"].nunique()
print(f"\n[5] Tramos cuya longitud cambia entre registros: {(var > 1).sum()} de {len(var)} "
      "-> se usa la mediana.")

# --------------------------------------------------------------------------- #
# Parte 2 — Ejecutar la limpieza
# --------------------------------------------------------------------------- #
titulo("PARTE 2 — Ejecutando la estandarización de todas las fuentes")
resumen = fuentes.estandarizar_todo()
for k, val in resumen.items():
    print(f"  {k:<26} {val}")

# --------------------------------------------------------------------------- #
# Parte 3 — Resultado
# --------------------------------------------------------------------------- #
titulo("PARTE 3 — Tablas limpias en data/interim/")
sens = pd.read_parquet(config.SENSORES_STD)
print(f"Velocidad horaria: {len(sens):,} filas (antes {len(raw):,}), "
      f"{sens['tramo_id'].nunique()} tramos, {sens['timestamp'].min()} a {sens['timestamp'].max()}")
print(f"Duplicados (tramo, hora) tras limpiar: {sens.duplicated(['tramo_id', 'timestamp']).sum()}")
print(f"Lecturas a las :30 tras limpiar: {(sens['timestamp'].dt.minute != 0).sum()}")
print("\nPrimeras filas:")
print(sens.head().to_string(index=False))

print("\nCatálogo de tramos:")
tr = pd.read_csv(config.TRAMOS_STD)
print(tr[["tramo_id", "corredor", "sentido", "inicio", "fin", "longitud_km", "flujo_tipo", "macrozona"]]
      .to_string(index=False))

print("\nLecturas por año:")
print(sens.groupby(sens["timestamp"].dt.year).size().to_string())

eod = pd.read_csv(config.EOD_HORA_STD)
od = pd.read_csv(config.EOD_OD_STD)
print(f"\nEOD 2025: matriz OD con {len(od):,} pares que suman {od['viajes'].sum():,.0f} viajes.")
print("Viajes por hora (6-9 h):")
print(eod.loc[eod["hora"].between(6, 9), ["hora", "viajes_publico", "viajes_privado", "viajes_no_motorizado",
                                          "viajes_total"]].round(0).to_string(index=False))

clima = pd.read_parquet(config.CLIMA_STD)
print(f"\nClima: {len(clima):,} horas, {(clima['lluvia_mm'] > 0).mean():.1%} con lluvia.")

print("\nArchivos generados:")
for p in sorted(config.DATA_INTERIM.glob("*")):
    if p.name != ".gitkeep":
        print(f"  {p.relative_to(config.ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")
