"""Descarga todas las fuentes reales a data/raw/ sin entrenar (multiplataforma).

Uso:  python scripts/descargar_datos.py [--forzar]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobipredict.data_acquisition import FuenteNoDisponible, adquirir_datos  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forzar", action="store_true", help="volver a descargar aunque existan")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        m = adquirir_datos(forzar=a.forzar)
    except FuenteNoDisponible as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    print(json.dumps({k: {"origen": v["origen"], "MB": round(v["bytes"] / 1e6, 1)} for k, v in m["fuentes"].items()},
                     indent=2, ensure_ascii=False))
    if m["errores"]:
        print("Fuentes con error (opcionales):", m["errores"])
