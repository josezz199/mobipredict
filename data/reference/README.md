# Tablas de referencia (versionadas)

Pequeñas tablas extraídas de fuentes oficiales que no se publican como archivo descargable.

| Archivo | Contenido | Origen | Fecha de extracción |
| --- | --- | --- | --- |
| `eod2025_macrozonas.csv` | Las 67 macrozonas de la EOD 2025 (id usado por la matriz OD de la API, municipio, macrozona, tipo) y su centroide | Diccionario `Ii` del visor oficial https://eod2025.metropol.gov.co/ y polígonos `macrozonas_actualizado.geo.json` del mismo visor. Centroide = centroide del polígono exterior en grados (aproximación plana) | 2026-10-23 |
| `corredores_simm.csv` | Ubicación media de las cámaras CCTV/ARS del SIMM por corredor (WGS84) | Columna `LOCATION` del conjunto MEData *Conteo Vehicular (Simmtrafficdata)*, enero de 2021 (https://medata.app.medellin.gov.co/sites/default/files/distribution/1-023-25-000300/simmtrafficdata.csv). El nombre `San Juan` del SIMM se renombró `Avenida San Juan` para casar con los tramos de velocidad | 2026-10-23 |

Si el AMVA o la Alcaldía cambian estos identificadores, regenere las tablas desde las mismas fuentes.
