/* MobiPredict — frontend sin dependencias. Consume la API de la misma aplicación. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const API = "";   // misma URL que el frontend

const ESTADO = {
  tramos: new Map(),      // tramo_id -> info del catálogo
  escenario: null,        // escenario actual (lecturas reales)
  predicciones: new Map(),// tramo_id -> respuesta de /predict
  seleccionado: null,
  editado: false,
};

const NOMBRES_MODELO = {
  perfil_historico: "Perfil histórico (línea base)",
  regla_umbral: "Regla simple (línea base)",
  ridge: "Ridge (lineal)",
  logistica: "Regresión logística",
  random_forest: "Random Forest",
  lightgbm: "LightGBM",
  mlp: "Red neuronal (MLP)",
};

const FIGURAS = {
  "01_congestion_cronica_heatmap.png": ["Congestión por vía y hora", "Más oscuro = más lento que lo normal. Hacia el centro en la mañana, de regreso en la tarde."],
  "02_eod2025_viajes_por_hora.png": ["Viajes por hora (EOD 2025)", "Dos picos: salida a las 6 h y regreso a las 17 h."],
  "03_eod2025_reparto_modal.png": ["En qué se mueve la gente", "A pie es el primer modo y la moto ya supera al carro."],
  "04_eod2025_densidad_destinos.png": ["Zonas que más viajes atraen", "El centro y El Poblado concentran los destinos; ahí están los sensores."],
  "05_siniestralidad_vial.png": ["Siniestralidad vial", "Incidentes fatales por año y todos los incidentes por hora del día."],
  "06_horas_en_colapso_por_tramo.png": ["Horas en colapso por vía", "Porcentaje del tiempo laborable por debajo de la mitad de la velocidad normal."],
  "07_drift_pandemia.png": ["El efecto de la pandemia", "La congestión cayó casi a cero en la cuarentena y luego se recuperó."],
  "08_lluvia_vs_colapso.png": ["Lluvia y colapso", "Con lluvia fuerte el colapso en hora pico es mucho más frecuente."],
  "09_composicion_vehicular_aforos.png": ["Qué vehículos circulan", "Proporción de carros, motos y buses contados por hora."],
};

const DIAS = ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"];
const MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];

/* ---------------- utilidades ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function num(x, dec = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toLocaleString("es-CO", { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function pct(x, dec = 0) { return x == null ? "—" : `${num(x * 100, dec)} %`; }
function fechaLocal(iso) {           // "2020-02-27T17:00:00" -> Date local sin conversión de zona
  const [f, h = "00:00"] = iso.split("T");
  const [a, m, d] = f.split("-").map(Number);
  const [hh, mm] = h.split(":").map(Number);
  return new Date(a, m - 1, d, hh, mm);
}
function hora(iso) { const d = fechaLocal(iso); return `${String(d.getHours()).padStart(2, "0")}:00`; }
function fechaLarga(iso) {
  const d = fechaLocal(iso);
  const dia = DIAS[d.getDay()];
  return `${dia.charAt(0).toUpperCase()}${dia.slice(1)} ${d.getDate()} de ${MESES[d.getMonth()]} de ${d.getFullYear()}`;
}
function fechaCorta(iso) { const d = fechaLocal(iso); return `${d.getDate()} ${MESES[d.getMonth()].slice(0, 3)} ${d.getFullYear()}`; }

async function pedir(ruta, opciones = {}) {
  const r = await fetch(API + ruta, { headers: { "Content-Type": "application/json" }, ...opciones });
  let cuerpo = null;
  try { cuerpo = await r.json(); } catch { /* respuesta sin JSON */ }
  if (!r.ok) {
    const det = cuerpo && cuerpo.detail ? (typeof cuerpo.detail === "string" ? cuerpo.detail : JSON.stringify(cuerpo.detail)) : `HTTP ${r.status}`;
    const e = new Error(det); e.status = r.status; throw e;
  }
  return cuerpo;
}

const etiquetaNivel = { normal: "Normal", precaucion: "Precaución", colapso: "Colapso" };
const sentidoTexto = { NS: "norte → sur", SN: "sur → norte", EW: "oriente → occidente", WE: "occidente → oriente" };

/* ---------------- arranque ---------------- */
async function iniciar() {
  cargarGaleria();
  let salud;
  try {
    salud = await pedir("/health");
  } catch (e) {
    $("#estado").textContent = "Sin conexión con la API";
    $("#estado").className = "estado mal";
    mostrarSinModelos(`No se pudo consultar /health: ${e.message}`);
    return;
  }
  if (!salud.modelos_cargados) {
    $("#estado").textContent = "API activa, sin modelos";
    $("#estado").className = "estado mal";
    mostrarSinModelos(salud.detalle || "");
    return;
  }
  $("#estado").textContent = `Modelos listos: ${NOMBRES_MODELO[salud.modelo_velocidad] || salud.modelo_velocidad} y ${NOMBRES_MODELO[salud.modelo_colapso] || salud.modelo_colapso}`;
  $("#estado").className = "estado ok";

  try {
    const tramos = await pedir("/tramos");
    tramos.forEach((t) => ESTADO.tramos.set(t.tramo_id, t));
  } catch (e) {
    mostrarSinModelos(`No se pudo cargar el catálogo de corredores: ${e.message}`);
    return;
  }
  $("#tablero").hidden = false;
  cargarModelos();

  let escenarios = [];
  try { escenarios = await pedir("/escenarios"); } catch { escenarios = []; }
  if (!escenarios.length) {
    modoManual();
    return;
  }
  pintarMomentos(escenarios);
  const inicial = escenarios.find((e) => e.id === "pico") || escenarios[0];
  elegirMomento(inicial.id);
}

function mostrarSinModelos(detalle) {
  $("#sin-modelos").hidden = false;
  $("#sin-modelos-detalle").textContent = detalle;
}

/* ---------------- momentos (escenarios reales) ---------------- */
function pintarMomentos(escenarios) {
  const cont = $("#momentos");
  cont.innerHTML = "";
  escenarios.forEach((e) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "momento" + (e.periodo === "drift" ? " drift" : "");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", "false");
    b.dataset.id = e.id;
    b.innerHTML = `${esc(e.nombre)}<small>${esc(fechaCorta(e.timestamp))}, ${hora(e.timestamp)} → ${hora(e.timestamp_prediccion)}</small>`;
    b.addEventListener("click", () => elegirMomento(e.id));
    cont.appendChild(b);
  });
}

async function elegirMomento(id) {
  document.querySelectorAll(".momento").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.id === id)));
  const elegido = document.querySelector(`.momento[data-id="${CSS.escape(id)}"]`);
  const cont = $("#momentos");
  if (elegido && cont.scrollWidth > cont.clientWidth) cont.scrollLeft = elegido.offsetLeft - cont.offsetLeft - 16;
  $("#resumen").innerHTML = `<p class="dato">Calculando predicciones…</p>`;
  try {
    const esc_ = await pedir(`/escenarios/${encodeURIComponent(id)}`);
    ESTADO.escenario = esc_;
    const peticiones = esc_.tramos.map((t) => ({ tramo_id: t.tramo_id, observaciones: t.observaciones }));
    const r = await pedir("/predict/batch", { method: "POST", body: JSON.stringify({ peticiones }) });
    ESTADO.predicciones = new Map(r.predicciones.map((p) => [p.tramo_id, p]));
    pintarResumen();
    pintarVias();
    const sel = ESTADO.seleccionado && ESTADO.predicciones.has(ESTADO.seleccionado)
      ? ESTADO.seleccionado
      : [...ESTADO.predicciones.values()].sort((a, b) => b.probabilidad_colapso - a.probabilidad_colapso)[0]?.tramo_id;
    if (sel) seleccionarVia(sel);
  } catch (e) {
    $("#resumen").innerHTML = `<p class="error">No se pudo calcular este momento: ${esc(e.message)}</p>`;
  }
}

function datosReales(tramoId) {
  return ESTADO.escenario?.tramos.find((t) => t.tramo_id === tramoId) || null;
}

function pintarResumen() {
  const e = ESTADO.escenario;
  const preds = [...ESTADO.predicciones.values()];
  const alertas = preds.filter((p) => p.alerta_colapso).length;
  const reales = e.tramos.filter((t) => t.colapso_real).length;
  const aciertos = e.tramos.filter((t) => ESTADO.predicciones.get(t.tramo_id)?.alerta_colapso === t.colapso_real).length;
  const errores = e.tramos.map((t) => Math.abs((ESTADO.predicciones.get(t.tramo_id)?.velocidad_predicha_kmh ?? NaN) - t.velocidad_real_kmh)).filter((x) => !Number.isNaN(x));
  const mae = errores.reduce((a, b) => a + b, 0) / Math.max(errores.length, 1);
  $("#resumen").innerHTML = `
    <p class="cuando">${esc(fechaLarga(e.timestamp))}, ${hora(e.timestamp)}. Predicción para las ${hora(e.timestamp_prediccion)}.</p>
    <p class="dato ${alertas ? "alerta" : ""}"><b>${alertas}</b>de ${preds.length} vías con alerta de colapso</p>
    <p class="dato"><b>${reales}</b>colapsaron de verdad</p>
    <p class="dato"><b>${aciertos}/${e.tramos.length}</b>vías con la alerta acertada</p>
    <p class="dato"><b>${num(mae)}</b>km/h de error medio en la velocidad</p>
    ${e.periodo === "drift" ? `<p class="aviso-drift">Este momento es de la cuarentena de 2020: el modelo nunca vio un tráfico así. Sirve para ver cómo se comporta cuando el mundo cambia.</p>` : ""}`;
}

function pintarVias() {
  const lista = $("#lista-vias");
  lista.innerHTML = "";
  const preds = [...ESTADO.predicciones.values()].sort((a, b) => b.probabilidad_colapso - a.probabilidad_colapso);
  preds.forEach((p) => {
    const real = datosReales(p.tramo_id);
    const ahora = real ? real.observaciones.at(-1).velocidad_kmh : null;
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.className = "via";
    b.dataset.id = p.tramo_id;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `
      <span class="via-nombre">${esc(p.corredor)}<span class="sentido" title="${esc(sentidoTexto[p.sentido] || p.sentido)}">${esc(p.sentido)}</span>
        <small>${esc(ESTADO.tramos.get(p.tramo_id)?.inicio || "")} a ${esc(ESTADO.tramos.get(p.tramo_id)?.fin || "")}</small></span>
      <span class="pista-wrap">${pista(p, ahora, real?.velocidad_real_kmh)}
        <span class="cifras">ahora <b>${num(ahora)}</b> → en 1 h <b>${num(p.velocidad_predicha_kmh)}</b> km/h${real ? `, real ${num(real.velocidad_real_kmh)}` : ""}, normal ${num(p.velocidad_referencia_kmh, 0)}</span></span>
      <span><span class="riesgo ${p.nivel}">${etiquetaNivel[p.nivel]} ${pct(p.probabilidad_colapso)}</span>
        ${real ? `<span class="acierto">${real.colapso_real ? "Colapsó" : "No colapsó"}${(real.colapso_real === p.alerta_colapso) ? ", acierto" : ", fallo"}</span>` : ""}</span>`;
    b.addEventListener("click", () => seleccionarVia(p.tramo_id));
    li.appendChild(b);
    lista.appendChild(li);
  });
}

function pista(p, ahora, real) {
  const ref = p.velocidad_referencia_kmh;
  const max = Math.max(ref * 1.15, ahora || 0, real || 0, p.velocidad_predicha_kmh) * 1.02;
  const x = (v) => `${Math.min(100, Math.max(0, (v / max) * 100))}%`;
  const umbral = ref * 0.5;
  const a = ahora ?? p.velocidad_predicha_kmh;
  const izq = Math.min(a, p.velocidad_predicha_kmh), der = Math.max(a, p.velocidad_predicha_kmh);
  return `<span class="pista" aria-hidden="true">
      <span class="carril"></span>
      <span class="zona" style="width:${x(umbral)}"></span>
      <span class="umbral" style="left:${x(umbral)}"></span>
      <span class="flecha" style="left:${x(izq)};width:calc(${x(der)} - ${x(izq)})"></span>
      ${ahora != null ? `<span class="marca-v ahora" style="left:${x(ahora)}"></span>` : ""}
      ${real != null ? `<span class="marca-v real" style="left:${x(real)}"></span>` : ""}
      <span class="marca-v pred" style="left:${x(p.velocidad_predicha_kmh)}"></span>
    </span>`;
}

/* ---------------- detalle de una vía ---------------- */
function seleccionarVia(tramoId) {
  ESTADO.seleccionado = tramoId;
  ESTADO.editado = false;
  document.querySelectorAll(".via").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === tramoId)));
  const real = datosReales(tramoId);
  const obs = real ? real.observaciones.map((o) => ({ ...o })) : lecturasManuales(tramoId);
  pintarDetalle(tramoId, obs, ESTADO.predicciones.get(tramoId) || null, real);
  if (!ESTADO.predicciones.get(tramoId)) predecirDetalle(tramoId, obs);
}

function lecturasManuales(tramoId) {
  const t = ESTADO.tramos.get(tramoId);
  const ref = t?.vel_referencia_kmh || 40;
  const base = new Date(2020, 1, 27, 14, 0);   // jueves laboral del periodo de prueba
  return [0.8, 0.72, 0.62, 0.5].map((f, i) => {
    const d = new Date(base.getTime() + i * 3600e3);
    const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}T${String(d.getHours()).padStart(2, "0")}:00:00`;
    return { timestamp: iso, velocidad_kmh: Math.round(ref * f * 10) / 10, lluvia_mm: 0 };
  });
}

function pintarDetalle(tramoId, obs, pred, real) {
  const t = ESTADO.tramos.get(tramoId) || {};
  const mostrarReal = real && !ESTADO.editado;
  const cont = $("#detalle");
  cont.innerHTML = `
    <h3>${esc(t.corredor || tramoId)} <span class="sentido">${esc(t.sentido || "")}</span></h3>
    <p class="ruta">${esc(t.inicio || "")} a ${esc(t.fin || "")}, ${num(t.longitud_km)} km, flujo ${esc((t.flujo_tipo || "").toLowerCase())}. Velocidad normal: ${num(t.vel_referencia_kmh, 0)} km/h.</p>
    <div class="grafico">${grafico(obs, pred, mostrarReal ? real.velocidad_real_kmh : null, t.vel_referencia_kmh)}</div>
    ${pred ? bloquePrediccion(pred, mostrarReal ? real : null) : `<p class="vacio">Calculando…</p>`}
    <details class="editar" ${real ? "" : "open"}>
      <summary>Probar otros valores</summary>
      <p class="prob-nota">Cambia la velocidad o la lluvia de las últimas horas y vuelve a predecir. Las horas deben ser consecutivas.</p>
      <table class="lecturas">
        <thead><tr><th>Hora</th><th>Velocidad (km/h)</th><th>Lluvia (mm)</th></tr></thead>
        <tbody>${obs.map((o, i) => `<tr>
          <td>${hora(o.timestamp)}</td>
          <td><input type="number" min="1" max="130" step="0.1" value="${o.velocidad_kmh}" data-i="${i}" data-c="velocidad_kmh" aria-label="Velocidad a las ${hora(o.timestamp)}"></td>
          <td><input type="number" min="0" max="100" step="0.1" value="${o.lluvia_mm}" data-i="${i}" data-c="lluvia_mm" aria-label="Lluvia a las ${hora(o.timestamp)}"></td>
        </tr>`).join("")}</tbody>
      </table>
      <button type="button" class="boton" id="btn-predecir">Predecir con estos valores</button>
      ${real ? `<button type="button" class="boton sec" id="btn-restaurar">Volver a lo medido</button>` : ""}
      <p class="error" id="error-detalle" role="alert"></p>
    </details>`;
  $("#btn-predecir").addEventListener("click", () => {
    const nuevas = obs.map((o) => ({ ...o }));
    let invalido = false;
    cont.querySelectorAll(".lecturas input").forEach((inp) => {
      const v = Number(inp.value);
      if (inp.value === "" || Number.isNaN(v) || v < Number(inp.min) || v > Number(inp.max)) invalido = true;
      nuevas[Number(inp.dataset.i)][inp.dataset.c] = v;
    });
    if (invalido) { $("#error-detalle").textContent = "Revisa los valores: la velocidad va de 1 a 130 km/h y la lluvia de 0 a 100 mm."; return; }
    ESTADO.editado = true;
    predecirDetalle(tramoId, nuevas);
  });
  $("#btn-restaurar")?.addEventListener("click", () => seleccionarVia(tramoId));
}

async function predecirDetalle(tramoId, obs) {
  try {
    const p = await pedir("/predict", { method: "POST", body: JSON.stringify({ tramo_id: tramoId, observaciones: obs }) });
    pintarDetalle(tramoId, obs, p, datosReales(tramoId));
    if (ESTADO.editado) $("#detalle details.editar").open = true;
  } catch (e) {
    const el = $("#error-detalle");
    if (el) el.textContent = `No se pudo predecir: ${e.message}`;
  }
}

function bloquePrediccion(p, real) {
  const u = p.umbral_decision;
  const texto = {
    normal: "El modelo no espera colapso en la próxima hora.",
    precaucion: "La probabilidad se acerca al umbral de alerta: conviene vigilar este corredor.",
    colapso: "El modelo emite alerta: espera que la vía vaya a menos de la mitad de su velocidad normal.",
  }[p.nivel];
  return `
    <div class="metricas">
      <div class="metrica"><span>Velocidad en 1 h</span><b>${num(p.velocidad_predicha_kmh)}</b> km/h</div>
      <div class="metrica"><span>${real ? "Velocidad real" : "Tiempo de viaje"}</span><b>${real ? num(real.velocidad_real_kmh) : num(p.tiempo_viaje_predicho_min)}</b> ${real ? "km/h" : "min"}</div>
      <div class="metrica"><span>${real ? "Tiempo de viaje" : "Velocidad normal"}</span><b>${real ? num(p.tiempo_viaje_predicho_min) : num(p.velocidad_referencia_kmh, 0)}</b> ${real ? "min" : "km/h"}</div>
    </div>
    <div class="prob">
      <div class="prob-txt"><span>Probabilidad de colapso a las ${hora(p.timestamp_prediccion)}</span><b><span class="riesgo ${p.nivel}">${etiquetaNivel[p.nivel]} ${pct(p.probabilidad_colapso)}</span></b></div>
      <div class="prob-barra" role="img" aria-label="Probabilidad ${pct(p.probabilidad_colapso)}, umbral de alerta ${pct(u)}">
        <span class="prob-relleno" style="width:${p.probabilidad_colapso * 100}%"></span>
        <span class="prob-corte" style="left:calc(${u * 100}% - 1px)" title="Umbral de alerta"></span>
      </div>
      <p class="prob-nota">${texto} La línea roja marca el umbral de alerta (${pct(u)}), elegido en validación.${real ? ` En la realidad, la vía ${real.colapso_real ? "sí colapsó" : "no colapsó"}.` : ""}</p>
    </div>`;
}

function grafico(obs, pred, real, ref) {
  const W = 460, H = 220, m = { l: 40, r: 16, t: 14, b: 30 };
  const puntos = obs.map((o) => o.velocidad_kmh);
  const n = obs.length + 1;   // + hora predicha
  const max = Math.max(ref || 0, ...puntos, pred?.velocidad_predicha_kmh || 0, real || 0) * 1.12;
  const x = (i) => m.l + (i * (W - m.l - m.r)) / (n - 1);
  const y = (v) => H - m.b - (v / max) * (H - m.t - m.b);
  const umbral = (ref || 0) * 0.5;
  const pasos = [0, max / 2, max].map((v) => Math.round(v / 5) * 5);
  const horas = obs.map((o) => hora(o.timestamp)).concat(pred ? hora(pred.timestamp_prediccion) : "+1 h");
  const linea = obs.map((o, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(o.velocidad_kmh).toFixed(1)}`).join(" ");
  const ult = obs.length - 1;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Velocidad de las últimas horas y predicción para la siguiente">
    ${pasos.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#e3e7e3"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("")}
    <rect x="${m.l}" y="${y(umbral)}" width="${W - m.l - m.r}" height="${H - m.b - y(umbral)}" fill="#f8dedb" opacity=".6"/>
    <line x1="${m.l}" x2="${W - m.r}" y1="${y(umbral)}" y2="${y(umbral)}" stroke="#c8372d" stroke-width="1.5"/>
    <text x="${m.l + 6}" y="${y(umbral) + 14}" style="fill:#c8372d">zona de colapso (&lt; ${num(umbral, 0)} km/h)</text>
    ${ref ? `<line x1="${m.l}" x2="${W - m.r}" y1="${y(ref)}" y2="${y(ref)}" stroke="#56636c" stroke-dasharray="4 4"/>
    <text x="${m.l + 6}" y="${y(ref) - 6}">velocidad normal ${num(ref, 0)} km/h</text>` : ""}
    <path d="${linea}" fill="none" stroke="#22303a" stroke-width="2.5" stroke-linejoin="round"/>
    ${obs.map((o, i) => `<circle cx="${x(i)}" cy="${y(o.velocidad_kmh)}" r="3.5" fill="#22303a"/>`).join("")}
    ${pred ? `<path d="M${x(ult)} ${y(obs[ult].velocidad_kmh)} L${x(n - 1)} ${y(pred.velocidad_predicha_kmh)}" stroke="#2b67b1" stroke-width="2" stroke-dasharray="5 4" fill="none"/>
      <circle cx="${x(n - 1)}" cy="${y(pred.velocidad_predicha_kmh)}" r="7" fill="#2b67b1" stroke="#fff" stroke-width="2"/>` : ""}
    ${real != null ? `<circle cx="${x(n - 1)}" cy="${y(real)}" r="9" fill="none" stroke="#22303a" stroke-width="2"/>` : ""}
    ${horas.map((h, i) => `<text x="${x(i)}" y="${H - 10}" text-anchor="middle" ${i === n - 1 ? 'style="fill:#2b67b1;font-weight:600"' : ""}>${h}</text>`).join("")}
  </svg>`;
}

/* ---------------- sin escenarios: modo manual ---------------- */
function modoManual() {
  $("#momentos").innerHTML = "";
  $("#resumen").innerHTML = `<p class="dato">No se encontraron momentos reales (models/escenarios.json). Elige un corredor y escribe sus últimas lecturas para predecir.</p>`;
  const lista = $("#lista-vias");
  lista.innerHTML = "";
  [...ESTADO.tramos.values()].forEach((t) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.className = "via"; b.dataset.id = t.tramo_id; b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span class="via-nombre">${esc(t.corredor)}<span class="sentido">${esc(t.sentido)}</span><small>${esc(t.inicio || "")} a ${esc(t.fin || "")}</small></span><span class="cifras">velocidad normal ${num(t.vel_referencia_kmh, 0)} km/h</span><span></span>`;
    b.addEventListener("click", () => seleccionarVia(t.tramo_id));
    li.appendChild(b); lista.appendChild(li);
  });
}

/* ---------------- modelos ---------------- */
async function cargarModelos() {
  let d;
  try { d = await pedir("/modelos"); } catch (e) {
    $("#ganadores").innerHTML = `<p class="error">No se pudo cargar la información de los modelos: ${esc(e.message)}</p>`;
    return;
  }
  const v = d.modelo_velocidad || {}, c = d.modelo_colapso || {};
  $("#ganadores").innerHTML = `
    <article class="ganador">
      <p class="tarea">Velocidad dentro de 1 hora</p>
      <h3>${esc(NOMBRES_MODELO[v.nombre] || v.nombre)}</h3>
      <div class="nums">
        <div><span>Error medio (MAE) en prueba</span><b>${num(v.test_mae_kmh, 2)} km/h</b></div>
        <div><span>Error relativo (WAPE)</span><b>${pct(v.test_wape, 1)}</b></div>
      </div>
      <ul>${(v.razones || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    </article>
    <article class="ganador col">
      <p class="tarea">Alerta de colapso dentro de 1 hora</p>
      <h3>${esc(NOMBRES_MODELO[c.nombre] || c.nombre)}</h3>
      <div class="nums">
        <div><span>Colapsos detectados (recall)</span><b>${pct(c.test_recall)}</b></div>
        <div><span>Alertas correctas (precisión)</span><b>${pct(c.test_precision)}</b></div>
        <div><span>PR-AUC en prueba</span><b>${num(c.test_pr_auc, 3)}</b></div>
      </div>
      <ul>${(c.razones || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    </article>`;

  const filas = d.comparacion || [];
  const tabla = (sel, tarea, cols) => {
    const rows = filas.filter((f) => f.tarea === tarea);
    $(sel).innerHTML = `<thead><tr><th>Modelo</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((f) => `<tr class="${f.seleccionado ? "elegido" : ""}"><td>${esc(NOMBRES_MODELO[f.modelo] || f.modelo)}</td>${cols.map((c) => `<td>${c[2](f[c[0]])}</td>`).join("")}</tr>`).join("")}</tbody>`;
  };
  const kmh = (x) => num(x, 2), p1 = (x) => pct(x, 1), p3 = (x) => num(x, 3), ms = (x) => num(x, 1);
  tabla("#tabla-velocidad", "velocidad", [
    ["test_mae", "MAE prueba (km/h)", kmh], ["test_rmse", "RMSE prueba", kmh], ["test_wape", "WAPE prueba", p1],
    ["test_mae_pico", "MAE en hora pico", kmh], ["drift_mae", "MAE pandemia", kmh], ["latencia_ms_1000", "ms por 1.000 filas", ms],
  ]);
  tabla("#tabla-colapso", "colapso", [
    ["test_pr_auc", "PR-AUC prueba", p3], ["test_recall_colapso", "Recall prueba", p1], ["test_precision_colapso", "Precisión prueba", p1],
    ["test_falsas_alarmas_por_dia", "Falsas alarmas por día", (x) => num(x, 1)], ["drift_pr_auc", "PR-AUC pandemia", p3], ["latencia_ms_1000", "ms por 1.000 filas", ms],
  ]);
}

/* ---------------- galería EDA ---------------- */
function cargarGaleria() {
  const g = $("#galeria");
  Object.entries(FIGURAS).forEach(([archivo, [titulo, texto]]) => {
    const fig = document.createElement("figure");
    fig.className = "figura";
    fig.innerHTML = `<button type="button" aria-label="Ampliar: ${esc(titulo)}"><img src="/figuras/${archivo}" alt="${esc(titulo)}" loading="lazy"></button>
      <figcaption><b>${esc(titulo)}</b>${esc(texto)}</figcaption>`;
    fig.querySelector("img").addEventListener("error", () => fig.remove());
    fig.querySelector("button").addEventListener("click", () => {
      $("#visor-img").src = `/figuras/${archivo}`;
      $("#visor-img").alt = titulo;
      $("#visor-texto").textContent = `${titulo}. ${texto}`;
      $("#visor").showModal();
    });
    g.appendChild(fig);
  });
  setTimeout(() => {
    if (!g.children.length) g.innerHTML = `<p class="nota">Aún no hay figuras. Se generan al ejecutar <code>python run_pipeline.py</code>.</p>`;
  }, 2500);
}

document.addEventListener("DOMContentLoaded", iniciar);
