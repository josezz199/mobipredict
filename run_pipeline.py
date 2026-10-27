"""Punto de entrada único del pipeline MobiPredict.

Uso:
    python run_pipeline.py              # corrida completa
    python run_pipeline.py --fast       # corrida rápida (menos trials de búsqueda)
    python run_pipeline.py --skip-eda   # omite el EDA
    python run_pipeline.py --descargar  # vuelve a descargar las fuentes aunque existan en data/raw/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline MLOps de MobiPredict (Prefect + MLflow)")
    p.add_argument("--fast", action="store_true", help="modo rápido para pruebas")
    p.add_argument("--skip-eda", action="store_true", help="no ejecutar el EDA")
    p.add_argument("--descargar", action="store_true", help="forzar nueva descarga de las fuentes")
    a = p.parse_args()
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")   # sin telemetría de Prefect
    if a.fast:
        os.environ["MOBIPREDICT_FAST"] = "1"   # debe fijarse antes de importar config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from mobipredict.pipeline import mobipredict_pipeline

    out = mobipredict_pipeline(ejecutar_eda=not a.skip_eda, forzar_descarga=a.descargar)
    print("\n=== Modelos seleccionados ===")
    print(json.dumps({k: out["metadata"][k] for k in ["modelo_velocidad", "modelo_colapso"]}, indent=2,
                     ensure_ascii=False))
    print("\nArtefactos: models/model_comparison.csv · docs/model_decision.md · mlruns/")


if __name__ == "__main__":
    main()
