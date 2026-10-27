# MobiPredict — Guía simple

Empieza aquí. Esta guía explica en palabras sencillas qué hace el proyecto, qué hay en cada carpeta y cómo ejecutarlo de principio a fin en Windows.

---

## 1. ¿Qué hace este proyecto?

Es un sistema de inteligencia artificial para el tráfico de Medellín. Hace tres cosas:

1. **Predice la velocidad** de 22 tramos de vías principales (Autopista Sur, Av. 80, Av. Regional…) para la **próxima hora**, y con eso el tiempo de viaje.
2. **Avisa si un tramo va a colapsar** en la próxima hora, es decir, si la velocidad va a caer a menos de la mitad de lo normal.
3. **Muestra cómo se mueve la gente** en el Valle de Aburrá según la Encuesta Origen-Destino 2025: 6,49 millones de viajes al día, a qué hora, en qué medio y hacia dónde.

Todo funciona con **datos reales y públicos** de la Alcaldía de Medellín, el Área Metropolitana y otras fuentes abiertas. Nada es inventado.

---

## 2. ¿Cómo funciona? (la idea en 6 pasos)

Al ejecutar `python run_pipeline.py` pasa esto, en orden y de forma automática:

```
1. Descarga los datos  →  2. Los limpia  →  3. Hace gráficos  →
4. Prepara los datos para el modelo  →  5. Entrena varios modelos  →  6. Elige el mejor
```

| Paso | En palabras simples |
| --- | --- |
| 1. Descargar | Baja de internet los archivos oficiales (velocidades, encuesta, accidentes, lluvia) |
| 2. Limpiar | Corrige errores de los datos (fechas mal escritas, registros repetidos) y los ordena |
| 3. Gráficos (EDA) | Dibuja gráficos para entender el tráfico: horas pico, zonas con más viajes, efecto de la lluvia… |
| 4. Preparar | Calcula la información que usa el modelo: la hora, la velocidad de las últimas horas, la lluvia, la demanda de viajes. Separa los datos en "para aprender" y "para evaluar" |
| 5. Entrenar | Prueba 9 modelos distintos y ajusta sus parámetros automáticamente. Guarda todo en MLflow, que es como un diario de experimentos |
| 6. Elegir | Compara los modelos con reglas claras, elige el mejor para cada tarea y explica por qué |

Después se puede encender una **API**, un servicio web al que le preguntas "¿cómo estará la Av. 80 en una hora?" y te responde.

---

## 3. ¿Qué hay en cada carpeta?

```
mobipredict/
├── run_pipeline.py      ← EL ARCHIVO PRINCIPAL: ejecuta todo el proceso
├── LEEME_PRIMERO.md     ← esta guía
├── README.md            ← resumen técnico
├── README_DETALLADO.md  ← explicación completa y detallada
│
├── src/mobipredict/     ← el código del proyecto (un archivo por paso)
├── api/                 ← el servicio web de predicción
├── scripts/             ← herramientas: descargar datos, regenerar el notebook
├── tests/               ← pruebas automáticas que verifican que todo funcione
├── notebooks/           ← todo el proyecto en un notebook de Jupyter
│
├── data/                ← los datos
│   ├── reference/       ← tablas oficiales que ya vienen incluidas
│   ├── raw/             ← aquí se guardan los datos descargados (empieza vacía)
│   ├── interim/         ← datos limpios (se llena solo)
│   └── processed/       ← datos listos para el modelo (se llena solo)
│
├── eda/figures/         ← los gráficos
├── eda/reports/         ← el reporte del análisis
├── models/              ← los modelos entrenados (se llena solo)
├── docs/                ← documentos: fuentes de datos y decisión del modelo
├── mlruns/              ← el "diario" de MLflow (se llena solo)
│
├── requirements.txt     ← lista de librerías necesarias
├── Dockerfile           ← receta para Docker
└── docker-compose.yml   ← enciende todo con Docker de una vez
```

### Los archivos del código (`src/mobipredict/`)

| Archivo | Qué hace (simple) |
| --- | --- |
| `config.py` | Todas las configuraciones en un solo lugar: enlaces de los datos, fechas, umbrales |
| `data_acquisition.py` | Paso 1: descarga los datos |
| `fuentes.py` | Paso 2: limpia y ordena los datos |
| `eda.py` | Paso 3: hace los gráficos y el reporte |
| `features.py` | Paso 4: prepara la información para el modelo |
| `training.py` | Paso 5: entrena los modelos y los registra en MLflow |
| `selection.py` | Paso 6: elige el mejor modelo y explica la decisión |
| `pipeline.py` | Une los 6 pasos en un solo flujo con Prefect |

### Los datos que usa

| Dato | De dónde viene | Para qué |
| --- | --- | --- |
| Velocidad por vía (2017-2020) | Alcaldía de Medellín (MEData) | Es lo que el modelo aprende a predecir |
| Encuesta Origen-Destino 2025 | Área Metropolitana (AMVA) | Saber cuánta gente viaja, a qué hora y entre qué zonas |
| Accidentes de tránsito (2008-2025) | Mendeley Data (registros de la Secretaría de Movilidad) | Gráficos de siniestralidad |
| Lluvia por hora | Open-Meteo | La lluvia empeora el tráfico |
| Conteo de vehículos | Alcaldía de Medellín | Saber cuántas motos y carros pasan |

La lista completa con enlaces está en `data/fuentes_datos.csv`.

---

## 4. Cómo ejecutarlo

Hay dos caminos. Elige **uno**:

- **Camino A — con Python.** Recomendado para aprender y explorar.
- **Camino B — con Docker.** Recomendado si solo quieres verlo funcionar.

### Antes de empezar (para ambos caminos)

1. Descomprime `mobipredict.zip` en una ruta corta, sin espacios ni tildes, y fuera de OneDrive. Por ejemplo: `C:\proyectos\mobipredict`.
2. Abre una terminal en esa carpeta: entra a la carpeta en el Explorador, escribe `powershell` en la barra de direcciones y pulsa Enter.
3. Necesitas **internet**: la primera vez se descargan unos 175 MB de datos.

---

### Camino A — Con Python

**Necesitas:** Python 3.11, 3.12 o 3.13 (recomendado 3.12). Verifica con `py --version`.

#### A1. Crear el entorno virtual

Es una "caja" aislada con las librerías de este proyecto.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

En **Git Bash** la activación es `source .venv/Scripts/activate`.

Debes ver `(.venv)` al inicio de la línea. Si aparece un error de "ejecución de scripts deshabilitada", ejecuta esto primero y vuelve a intentar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

> Cada vez que abras una terminal nueva, actívalo otra vez con `.venv\Scripts\Activate.ps1`.

#### A2. Instalar las librerías

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Tarda unos 5-10 minutos.

#### A3. Probar que todo quedó bien

```powershell
pytest -q tests
```

Debe decir algo como `5 passed, 5 skipped`. Las 5 "skipped" son las pruebas de la API, que se activan después de entrenar.

#### A4. Ejecutar el proyecto completo

```powershell
python run_pipeline.py --fast
```

Esto hace los 6 pasos: descarga, limpieza, gráficos, preparación, entrenamiento y selección.

- `--fast` tarda unos 5-10 minutos.
- Sin `--fast` hace una búsqueda más completa y tarda unos 15-25 minutos.

Verás mensajes de avance. Al final aparece algo así:

```
=== Modelos seleccionados ===
{ "modelo_velocidad": { "nombre": "...", "test_mae_kmh": ... },
  "modelo_colapso":   { "nombre": "...", "test_recall": ... } }
```

#### A5. Ver los resultados

| Qué ver | Dónde |
| --- | --- |
| Gráficos | Carpeta `eda\figures\` |
| Reporte del análisis | `eda\reports\eda_report.md` |
| Qué modelo ganó y por qué | `docs\model_decision.md` |
| Comparación de todos los modelos | `models\model_comparison.csv` (ábrelo con Excel) |

Para ver el "diario de experimentos" de MLflow en el navegador:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Abre **http://localhost:5000**. Ahí ves cada modelo probado, sus resultados y sus parámetros. Para cerrarlo: `Ctrl + C`.

#### A6. Encender la aplicación web y la API

En la terminal (con `(.venv)` activo):

```powershell
uvicorn api.main:app --port 8000
```

Abre **http://localhost:8000**. Es la aplicación de MobiPredict:

1. **Elige un momento** arriba (por ejemplo, "Pico de la tarde"). Son momentos reales del periodo de prueba.
2. Verás las **22 vías ordenadas por riesgo**: velocidad ahora, predicción a 1 hora, lo que de verdad pasó y si la alerta acertó.
3. **Haz clic en una vía** para ver su gráfico y su probabilidad de colapso. En "Probar otros valores" puedes cambiar la velocidad o la lluvia y volver a predecir.
4. Más abajo están **los modelos** (quién ganó y por qué) y **los gráficos del análisis de datos**.

Si prefieres probar la API directamente, abre **http://localhost:8000/docs**. Es una página donde puedes probar la API con botones:

1. Abre `GET /tramos` → **Try it out** → **Execute**: ves la lista de las 22 vías.
2. Abre `POST /predict` → **Try it out** → **Execute**: ya trae un ejemplo de la Av. 80 y te devuelve la predicción.

O desde otra terminal de PowerShell:

```powershell
$cuerpo = @{
  tramo_id = "AVENIDA_80_NS"
  observaciones = @(
    @{ timestamp = "2020-02-11T14:00:00"; velocidad_kmh = 24; lluvia_mm = 0 },
    @{ timestamp = "2020-02-11T15:00:00"; velocidad_kmh = 22; lluvia_mm = 0 },
    @{ timestamp = "2020-02-11T16:00:00"; velocidad_kmh = 19; lluvia_mm = 0.4 },
    @{ timestamp = "2020-02-11T17:00:00"; velocidad_kmh = 15; lluvia_mm = 1.2 }
  )
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $cuerpo -ContentType "application/json"
```

**Cómo leer la respuesta:**

| Campo | Significado |
| --- | --- |
| `velocidad_predicha_kmh` | Velocidad esperada dentro de una hora |
| `tiempo_viaje_predicho_min` | Minutos para recorrer el tramo |
| `probabilidad_colapso` | Probabilidad de colapso, de 0 a 1 |
| `alerta_colapso` | `True` si el sistema recomienda alertar |
| `nivel` | `normal`, `precaucion` o `colapso` |

Para apagar la API: `Ctrl + C`.

---

### Camino B — Con Docker

**Necesitas:** Docker Desktop instalado y **abierto**. Verifica con `docker --version`.

#### B1. Encender todo

```powershell
docker compose up --build
```

Esto hace, solo:

1. Construye el contenedor con Python y las librerías (la primera vez tarda 5-10 minutos).
2. Descarga los datos y ejecuta los 6 pasos (servicio `trainer`).
3. Enciende la aplicación web en **http://localhost:8000** (y la API en **/docs**).
4. Enciende MLflow en **http://localhost:5000**.

Para un entrenamiento rápido:

```powershell
$env:MOBIPREDICT_FAST="1"; docker compose up --build
```

#### B2. Ver resultados

Los resultados aparecen en tus carpetas normales de Windows (`eda\`, `docs\`, `models\`), igual que en el camino A. La API se usa igual que en el paso A6.

#### B3. Apagar

`Ctrl + C` y luego:

```powershell
docker compose down
```

---

## 5. Otras formas de ejecutarlo

- **Con el notebook:** abre `notebooks\MobiPredict_MLOps_end_to_end.ipynb` en VS Code o Jupyter y ejecuta todas las celdas. Hace todo lo anterior y muestra los resultados en pantalla. También funciona en Google Colab: súbelo y dale "Ejecutar todo".
- **Solo descargar los datos**, sin entrenar:

  ```powershell
  python scripts\descargar_datos.py
  ```

---

## 6. Problemas comunes

| Problema | Solución |
| --- | --- |
| "La ejecución de scripts está deshabilitada" | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `python` no se reconoce o abre la Microsoft Store | Usa `py -3.12` para crear el entorno; dentro del entorno activo `python` ya funciona |
| La página de http://localhost:8000 dice "Aún no hay modelos entrenados" | Ejecuta `python run_pipeline.py --fast` y recarga la página |
| Error instalando librerías con Python 3.14 | Usa Python 3.12 |
| "No se pudo descargar…" | Revisa tu internet y vuelve a ejecutar; lo ya descargado no se repite |
| La API dice "Faltan artefactos" | Primero ejecuta `python run_pipeline.py` |
| `mlflow ui` falla con "filesystem tracking backend" | Ejecuta antes `$env:MLFLOW_ALLOW_FILE_STORE="true"` |
| Docker no arranca | Abre Docker Desktop y espera a que diga "Engine running" |
| El puerto 8000 o 5000 está ocupado | Cambia el número: `--port 8001` o `--port 5001` |

---

## 7. Resumen de comandos (Camino A)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q tests
python run_pipeline.py --fast
uvicorn api.main:app --port 8000          # aplicación: http://localhost:8000 · API: /docs
```

## 8. ¿Quieres saber más?

- `README_DETALLADO.md`: cómo funciona cada parte en detalle.
- `docs\fuentes_de_datos.md`: de dónde salen los datos y cómo se limpiaron.
- `docs\model_decision.md`: qué modelo ganó y por qué (aparece después de ejecutar).

**Una limitación importante:** los datos públicos de velocidad llegan hasta 2020. El modelo aprende bien cómo se comporta el tráfico, pero para usarlo hoy habría que reentrenarlo con datos recientes de la Alcaldía.
