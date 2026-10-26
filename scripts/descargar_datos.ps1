# Descarga las fuentes reales de MobiPredict a data\raw\ (Windows PowerShell 5+ o PowerShell 7).
# Uso, desde la raíz del proyecto:
#   powershell -ExecutionPolicy Bypass -File scripts\descargar_datos.ps1
# Luego: python run_pipeline.py
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # acelera Invoke-WebRequest
$raw = Join-Path $PSScriptRoot "..\data\raw"
New-Item -ItemType Directory -Force -Path $raw | Out-Null

$fuentes = @(
  @{ nombre = "medata_velocidad_tiempo_viaje_gt.csv";
     url = "https://medata.app.medellin.gov.co/sites/default/files/distribution/1-023-25-000287/Velocidad_y_tiempo_de_viaje_GT.csv" },
  @{ nombre = "medata_aforos_vehiculares.csv";
     url = "https://medata.app.medellin.gov.co/sites/default/files/distribution/1-023-25-000301/Aforos_Vehiculares.csv" },
  @{ nombre = "mendeley_siniestros_medellin_2008_2025.zip";
     url = "https://data.mendeley.com/public-files/datasets/r6g5dfnpgh/files/e992c888-5819-40c6-9538-41980ea1d429/file_downloaded" },
  @{ nombre = "open_meteo_medellin_horario.json";
     url = "https://archive-api.open-meteo.com/v1/archive?latitude=6.2442&longitude=-75.5812&start_date=2017-01-01&end_date=2020-12-31&hourly=precipitation,temperature_2m&timezone=America%2FBogota" },
  @{ nombre = "eod2025_macrozonas.geo.json";
     url = "https://eod2025.metropol.gov.co/macrozonas_actualizado.geo.json" }
)

foreach ($f in $fuentes) {
  $destino = Join-Path $raw $f.nombre
  if (Test-Path $destino) { Write-Host "Ya existe: $($f.nombre)"; continue }
  Write-Host "Descargando $($f.nombre) ..."
  try {
    Invoke-WebRequest -Uri $f.url -OutFile $destino -UseBasicParsing -TimeoutSec 600
    $mb = [math]::Round((Get-Item $destino).Length / 1MB, 1)
    Write-Host "  OK ($mb MB)"
  } catch {
    Write-Warning "  No se pudo descargar $($f.url): $($_.Exception.Message)"
  }
}
Write-Host "`nLa EOD 2025 se descarga por API al correr el pipeline; si la API no responde se usa data\reference\eod2025."
Write-Host "Siguiente paso: python run_pipeline.py"
