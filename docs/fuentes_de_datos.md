# Fuentes de datos de MobiPredict

Todas las fuentes son abiertas y reales. No hay datos simulados en el pipeline. El inventario y la calidad se verificaron el 23 de octubre de 2026 leyendo los archivos completos.

## Resumen

| # | Fuente | Publica | Qué aporta al proyecto | Cobertura verificada | Formato y tamaño | Licencia |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [Velocidad y tiempo de viaje GT](https://www.datos.gov.co/d/7t5n-3b3w) | Secretaría de Movilidad de Medellín (SIMM), vía MEData | **Serie del modelo:** velocidad y tiempo de viaje por tramo | 22 tramos (11 corredores × 2 sentidos), 6:00-20:00, 1-feb-2017 a 15-dic-2020, 682.502 filas, 1.045 días | CSV, 89,8 MB | Datos abiertos Alcaldía de Medellín |
| 2 | [EOD 2025 — visor oficial](https://eod2025.metropol.gov.co/) | Área Metropolitana del Valle de Aburrá | Demanda: 6.490.299 viajes en día hábil, viajes por hora y modo, reparto modal, motorización, **matriz OD de 67 macrozonas** | Valle de Aburrá, 10 municipios, día hábil 2025 | API JSON pública (`/api/api/indicadores/{id}/agregado`) + GeoJSON de macrozonas | Datos públicos del AMVA |
| 3 | [Road-traffic incidents and fatalities in Medellín 2008-2025](https://doi.org/10.17632/r6g5dfnpgh.1) | Correa Álvarez et al. (2026), Mendeley Data, a partir de registros de la Secretaría de Movilidad | Siniestralidad georreferenciada | 702.540 incidentes, 1-ene-2008 a 31-ago-2025 (2.762 fatales) | ZIP con XLSX, 63 MB | CC BY 4.0 |
| 4 | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Open-Meteo (reanálisis ERA5 / ERA5-Land de Copernicus) | Lluvia y temperatura horaria en Medellín | 2017-2020, horario | API JSON | CC BY 4.0 |
| 5 | [Aforos vehiculares](https://www.datos.gov.co/d/b9s9-jw7c) | Secretaría de Movilidad de Medellín, vía MEData | Composición vehicular (≈ 40 % motos) por hora | 74.900 registros de 15 min, 2017-2019, hasta 113 intersecciones | CSV, 19,7 MB | Datos abiertos Alcaldía de Medellín |
| 6 | Calendario de festivos de Colombia | Paquete [`holidays`](https://pypi.org/project/holidays/) (Ley 51 de 1983) | Festivos para el tipo de día | 2017-2026 | Calculado en código | MIT |
| 7 | Cifras publicadas de referencia | Observatorio de Movilidad / AMVA, vía [El Colombiano](https://www.elcolombiano.com/medellin/balance-de-accidentalidad-en-medellin-el-ano-pasado-PF32374139) | 276 muertes viales en Medellín en 2025 (162 motociclistas, 100 peatones); +48 % tenencia de motos y −14 % participación del TP entre 2017 y 2025 ([nota](https://www.elcolombiano.com/medellin/movilidad-habitantes-medellin-aburra-disminucion-teletrabajo-DD36063145)) | 2025 | Texto | — |

URLs de descarga directa usadas por `src/mobipredict/data_acquisition.py`:

- Velocidad GT: `https://medata.app.medellin.gov.co/sites/default/files/distribution/1-023-25-000287/Velocidad_y_tiempo_de_viaje_GT.csv`
- Aforos: `https://medata.app.medellin.gov.co/sites/default/files/distribution/1-023-25-000301/Aforos_Vehiculares.csv`
- Siniestros: `https://data.mendeley.com/public-files/datasets/r6g5dfnpgh/files/e992c888-5819-40c6-9538-41980ea1d429/file_downloaded`
- EOD 2025: `https://eod2025.metropol.gov.co/api/api/metadata/indicadores`, `.../indicadores/{1..46}/agregado`, `https://eod2025.metropol.gov.co/macrozonas_actualizado.geo.json`
- Clima: `https://archive-api.open-meteo.com/v1/archive?latitude=6.2442&longitude=-75.5812&hourly=precipitation,temperature_2m&timezone=America/Bogota`

## Por qué estas y no otras

| Fuente revisada | Decisión | Motivo |
| --- | --- | --- |
| Velocidad y tiempo de viaje GT | **Elegida** como serie del modelo | Es la serie abierta más larga por corredor (casi 4 años), con resolución horaria y los corredores principales. Permite un split temporal con validación y prueba reales, y un periodo de drift (pandemia). |
| Velocidad e intensidad vehicular (SIMM, CCTV/ARS) | Descartada | Solo cubre julio y agosto de 2020 (431.490 filas), en plena pandemia. No representa la operación normal. |
| Conteo vehicular Simmtrafficdata | Usada solo para ubicar corredores | 855 MB, pero solo enero de 2021 (cada 5 min). Sirvió para obtener la ubicación real de las cámaras por corredor (`data/reference/corredores_simm.csv`). |
| Incidentes viales MEData | Reemplazada | Llega a 2020. La base de Mendeley (misma fuente administrativa) llega a agosto de 2025. |
| Microdatos EOD (2017 o 2025) | No disponibles | El portal de datos abiertos del AMVA respondió 403 a las consultas. Se usan los agregados oficiales de la API del visor EOD 2025, que incluyen la matriz OD por macrozona. |
| SIATA (lluvia por estación) | Alternativa futura | No ofrece una API pública documentada de series históricas. Open-Meteo da lluvia horaria reproducible. Si se obtiene la serie del SIATA, reemplaza a Open-Meteo sin cambiar el modelo. |

## Problemas de calidad encontrados y cómo se tratan

| Fuente | Problema | Tratamiento |
| --- | --- | --- |
| Velocidad GT | Las columnas `DÍA` y `MES` están desalineadas en parte de 2020 (p. ej. `FECHA_HORA` = 1/02/2020 con `DÍA` = 2). | La fecha se toma de `FECHA_HORA`, que coincide con `DIASEM` en el 100 % de las filas. Las filas incoherentes se descartarían (hoy: 0). |
| Velocidad GT | La resolución cambia de horaria a cada 30 minutos en 2020. | Todo se agrega al promedio horario. |
| Velocidad GT | 9.699 claves (tramo, fecha-hora) duplicadas. | Se promedian. |
| Velocidad GT | El ID de corredor 13/14 se usa a veces para Guayabal y para El Poblado; `INICIO`/`FIN` y la longitud varían levemente. | El tramo se identifica por nombre del corredor + sentido; la longitud es la mediana. |
| Velocidad GT | Encabezados y nombres de día con caracteres dañados (`Mi�rcoles`). | Lectura UTF-8 con reemplazo y columnas por posición. |
| Velocidad GT | No hay lecturas entre 21:00 y 5:59. | Los rezagos no cruzan la noche; no se imputa. |
| Velocidad GT | Marzo-diciembre de 2020 es un régimen distinto (pandemia). | Fuera de entrenamiento, validación y prueba; se usa como prueba de estrés de drift. |
| Siniestros | La base cuenta incidentes; el cierre oficial de 2025 cuenta personas fallecidas, y la base termina en agosto de 2025. | Se grafican incidentes fatales por año y se cita aparte el cierre oficial (276). |
| EOD 2025 | La API identifica las macrozonas solo con números. | El diccionario id → municipio/macrozona se extrajo del visor oficial (`data/reference/eod2025_macrozonas.csv`). |

## Limitación principal

La serie de velocidad abierta más reciente termina en diciembre de 2020 y la EOD es de 2025. El modelo aprende la dinámica de la congestión con datos 2017-2020 y usa la EOD 2025 como prior de demanda por hora y por macrozona. Para operar en 2026 hay que reentrenar con el feed actual del SIMM: basta con poner un CSV con las mismas columnas en `data/raw/` y correr `python run_pipeline.py`.
