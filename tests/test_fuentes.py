"""Pruebas de las fuentes: agregados oficiales de la EOD 2025 incluidos en data/reference/ y lectura
del ZIP real de siniestralidad (trae el diccionario y los datos con nombres parecidos)."""
import json

import pandas as pd

from mobipredict import config, fuentes


def test_eod_referencia_consistente():
    ref = config.DATA_REFERENCE / "eod2025"
    ind = json.loads((ref / "indicadores_api.json").read_text(encoding="utf-8"))["indicadores"]
    od = pd.read_csv(ref / "matriz_od_macrozonas.csv")
    assert round(od["viajes"].sum()) == round(ind["1"]["valor"]) == 6490299
    assert od["origen"].nunique() == 67
    mz = pd.read_csv(config.MACROZONAS_FILE)
    assert set(od["origen"]) <= set(mz["id_macrozona"])


def test_siniestros_elige_el_excel_de_datos(tmp_path, monkeypatch):
    """El ZIP real trae Data_Dictionary_Fatal_Road_Traffic.xlsx ANTES que Fatal_Road_Traffic.xlsx:
    debe leerse el de datos (hoja BD), no el diccionario."""
    import io
    import zipfile
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    raw.mkdir(); interim.mkdir()
    monkeypatch.setattr(config, "DATA_RAW", raw)
    monkeypatch.setattr(config, "SINIESTROS_STD", interim / "sin.parquet")

    def xlsx(df, hoja):
        b = io.BytesIO(); df.to_excel(b, sheet_name=hoja, index=False); return b.getvalue()

    datos = pd.DataFrame({"LLAVE": [1, 2], "AÑO": [2024, 2025], "FECHA_INCIDENTE": ["2024-05-01", "2025-01-02"],
                          "HORA_INCIDENTE": ["07:30:00", "18:10:00"], "CLASE_INCIDENTE": ["CHOQUE", "ATROPELLO"],
                          "GRAVEDAD_INCIDENTE": ["HERIDO", "MUERTO"], "Latitud": [6.25, 6.2],
                          "Longitud": [-75.57, -75.6], "Comuna Planeacion": ["La Candelaria", "Belén"]})
    carpeta = "Medellin_Road_Traffic_Fatalities_Dataset_2008_2025/"
    with zipfile.ZipFile(raw / config.FUENTES["siniestros"]["archivo"], "w") as z:
        z.writestr(carpeta + "Data_Dictionary_Fatal_Road_Traffic.xlsx",
                   xlsx(pd.DataFrame({"Variable": ["AÑO"]}), "Diccionario"))
        z.writestr(carpeta + "Fatal_Road_Traffic.xlsx", xlsx(datos, "BD"))

    out = fuentes.estandarizar_siniestros()
    assert len(out) == 2
    assert out["fatal"].tolist() == [0, 1]
    assert out["hora"].tolist() == [7, 18]
