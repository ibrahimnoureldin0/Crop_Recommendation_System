/* ══════════════════════════════════════════════════
   main.js  —  Crop Advisor Frontend Logic
   Communicates with Flask /api/predict endpoint
   ══════════════════════════════════════════════════ */

"use strict";

/* ── Field Definitions ─────────────────────────────────────────────────── */
const FIELDS = [
  { id: "N",           label: "النيتروجين (N)",     unit: "mg/kg", step: 1,   dp: 0 },
  { id: "P",           label: "الفوسفور (P)",        unit: "mg/kg", step: 1,   dp: 0 },
  { id: "K",           label: "البوتاسيوم (K)",      unit: "mg/kg", step: 1,   dp: 0 },
  { id: "temperature", label: "درجة الحرارة",        unit: "°C",    step: 0.1, dp: 1 },
  { id: "humidity",    label: "الرطوبة النسبية",     unit: "%",     step: 0.5, dp: 1 },
  { id: "ph",          label: "درجة الحموضة (pH)",   unit: "pH",    step: 0.1, dp: 1 },
  { id: "rainfall",    label: "الأمطار السنوية",     unit: "mm",    step: 1,   dp: 0 },
];

/* ── Preset values (mean of each crop from training data) ──────────────── */
const PRESETS = {
  rice:       { N:80,  P:48,  K:40,  temperature:23.7, humidity:82.3, ph:6.4,  rainfall:236 },
  banana:     { N:100, P:82,  K:50,  temperature:27.4, humidity:80.4, ph:5.98, rainfall:105 },
  mango:      { N:20,  P:27,  K:30,  temperature:31.2, humidity:50.2, ph:5.77, rainfall:95  },
  cotton:     { N:118, P:46,  K:20,  temperature:24.0, humidity:79.8, ph:6.91, rainfall:80  },
  coffee:     { N:101, P:29,  K:30,  temperature:25.5, humidity:58.9, ph:6.79, rainfall:158 },
  apple:      { N:21,  P:134, K:200, temperature:22.6, humidity:92.3, ph:5.93, rainfall:113 },
  watermelon: { N:99,  P:17,  K:50,  temperature:25.6, humidity:85.2, ph:6.50, rainfall:51  },
  grapes:     { N:23,  P:133, K:200, temperature:23.9, humidity:81.9, ph:6.03, rainfall:70  },
};

/* ── Build Input Fields ─────────────────────────────────────────────────── */
function buildInputs() {
  const grid = document.getElementById("inputs-grid");
  grid.innerHTML = "";

  FIELDS.forEach(f => {
    const r   = FEATURE_RANGES[f.id] || { min: 0, max: 100, mean: 50 };
    const def = parseFloat(r.mean).toFixed(f.dp);
    const pct = (((r.mean - r.min) / (r.max - r.min)) * 100).toFixed(1);

    grid.insertAdjacentHTML("beforeend", `
      <div class="field-group" id="grp_${f.id}">
        <div class="field-label">
          <span>${f.label}</span>
          <span class="field-unit">${f.unit}</span>
        </div>
        <div class="field-slider-row">
          <input type="range"
            id="sl_${f.id}"
            min="${r.min}" max="${r.max}" step="${f.step}"
            value="${def}"
            style="--pct:${pct}%"
            oninput="onSlider('${f.id}',this.value,${f.dp})">
          <div class="field-value-badge" id="badge_${f.id}">${def}</div>
        </div>
        <input type="number"
          class="number-input"
          id="num_${f.id}"
          min="${r.min}" max="${r.max}" step="${f.step}"
          value="${def}"
          oninput="onNumber('${f.id}',this.value,${f.dp})">
        <div class="range-hint">${r.min} — ${r.max}</div>
      </div>`);
  });
}

/* ── Sync helpers ───────────────────────────────────────────────────────── */
function onSlider(id, val, dp) {
  const r   = FEATURE_RANGES[id];
  const pct = (((val - r.min) / (r.max - r.min)) * 100).toFixed(1);
  document.getElementById(`sl_${id}`).style.setProperty("--pct", pct + "%");
  document.getElementById(`badge_${id}`).textContent = parseFloat(val).toFixed(dp);
  document.getElementById(`num_${id}`).value          = parseFloat(val).toFixed(dp);
}

function onNumber(id, val, dp) {
  const r   = FEATURE_RANGES[id];
  val = Math.min(r.max, Math.max(r.min, parseFloat(val) || r.min));
  const pct = (((val - r.min) / (r.max - r.min)) * 100).toFixed(1);
  document.getElementById(`sl_${id}`).value = val;
  document.getElementById(`sl_${id}`).style.setProperty("--pct", pct + "%");
  document.getElementById(`badge_${id}`).textContent = parseFloat(val).toFixed(dp);
}

/* ── Preset loader ──────────────────────────────────────────────────────── */
function loadPreset(crop) {
  const p = PRESETS[crop];
  if (!p) return;

  // Highlight active preset button
  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  const btn = document.querySelector(`.preset-btn[data-crop="${crop}"]`);
  if (btn) btn.classList.add("active");

  FIELDS.forEach(f => {
    const val = p[f.id] !== undefined ? p[f.id] : FEATURE_RANGES[f.id].mean;
    document.getElementById(`sl_${f.id}`).value = val;
    onSlider(f.id, val, f.dp);
    document.getElementById(`num_${f.id}`).value = parseFloat(val).toFixed(f.dp);
  });
}

/* ── Attach preset buttons ──────────────────────────────────────────────── */
function attachPresets() {
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => loadPreset(btn.dataset.crop));
  });
}

/* ── Collect current input values ───────────────────────────────────────── */
function collectInputs() {
  const params = {};
  FIELDS.forEach(f => {
    params[f.id] = parseFloat(document.getElementById(`num_${f.id}`).value);
  });
  return params;
}

/* ════════════════════════════════════════════════════
   API CALL  →  POST /api/predict
   ════════════════════════════════════════════════════ */
async function submitPrediction() {
  const btn       = document.getElementById("predict-btn");
  const btnText   = document.getElementById("btn-text");
  const btnLoader = document.getElementById("btn-loader");
  const errorBox  = document.getElementById("error-box");

  // Loading state
  btn.disabled = true;
  btnText.classList.add("hidden");
  btnLoader.classList.remove("hidden");
  errorBox.classList.add("hidden");

  try {
    const params   = collectInputs();
    const response = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(params),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const result = await response.json();
    renderResult(result);

  } catch (err) {
    errorBox.textContent = `⚠️ خطأ: ${err.message}`;
    errorBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
}

/* ── Render the prediction result ───────────────────────────────────────── */
function renderResult(data) {
  const panel = document.getElementById("result-panel");

  // Re-trigger animation
  panel.classList.add("hidden");
  void panel.offsetWidth;
  panel.classList.remove("hidden");

  /* ─ Hero section ─ */
  document.getElementById("result-name").textContent    = capitalize(data.crop);
  document.getElementById("result-name-ar").textContent = data.crop_ar || "";
  document.getElementById("crop-emoji").textContent     = data.emoji || "🌱";

  // Image
  const img = document.getElementById("crop-img");
  if (data.image_url) {
    img.src = data.image_url;
    img.alt = data.crop;
    img.onload  = () => img.classList.remove("hidden");
    img.onerror = () => img.classList.add("hidden");
  } else {
    img.classList.add("hidden");
  }

  // Confidence bar (animate after paint)
  const fill = document.getElementById("conf-fill");
  fill.style.width = "0%";
  requestAnimationFrame(() => setTimeout(() => {
    fill.style.width = Math.min(data.confidence, 99.9).toFixed(1) + "%";
  }, 80));
  document.getElementById("conf-text").textContent = data.confidence.toFixed(1) + "%";

  /* ─ Top 3 ─ */
  const t3 = document.getElementById("top3-container");
  t3.innerHTML = `<div class="top3-title">أعلى التوصيات</div>`;
  const maxConf = data.top_crops[0]?.confidence || 1;

  data.top_crops.forEach(item => {
    const barW = ((item.confidence / maxConf) * 100).toFixed(0);
    t3.insertAdjacentHTML("beforeend", `
      <div class="top3-item">
        <div class="top3-rank rank-${item.rank}">${item.rank}</div>
        <span class="top3-emoji">${item.emoji || "🌱"}</span>
        <span class="top3-name">
          ${capitalize(item.crop)}
          <small style="opacity:.5;font-weight:400">${item.ar || ""}</small>
        </span>
        <span class="top3-pct">${item.confidence.toFixed(1)}%</span>
        <div class="top3-bar-bg">
          <div class="top3-bar-fill" style="width:${barW}%"></div>
        </div>
      </div>`);
  });

  /* ─ Ideal conditions ─ */
  const sr       = document.getElementById("stats-row");
  const ideal    = data.ideal_conditions || {};
  const statDefs = [
    { key: "temperature", label: "درجة الحرارة المثالية", unit: "°C" },
    { key: "humidity",    label: "الرطوبة المثالية",      unit: "%"  },
    { key: "ph",          label: "درجة الحموضة",          unit: "pH" },
    { key: "rainfall",    label: "الأمطار الموصى بها",    unit: "mm" },
  ];
  sr.innerHTML = statDefs.map(s => {
    const val = ideal[s.key]?.mean ?? "—";
    return `<div class="stat-cell">
      <div class="stat-val">${typeof val === "number" ? val.toFixed(1) : val} ${s.unit}</div>
      <div class="stat-lbl">${s.label}</div>
    </div>`;
  }).join("");

  // Scroll into view
  setTimeout(() => panel.scrollIntoView({ behavior: "smooth", block: "nearest" }), 150);
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : "";
}

/* ── Wheat SVG animation ─────────────────────────────────────────────────── */
function buildWheat() {
  const wg = document.getElementById("wg");
  if (!wg) return;
  let html = "";
  for (let i = 0; i < 80; i++) {
    const x = i * 15 + 5;
    const h = 40 + Math.random() * 80;
    const delay = (Math.random() * 3).toFixed(2);
    const dur   = (2.5 + Math.random() * 2).toFixed(2);
    const rot   = (-15 + Math.random() * 30).toFixed(1);
    const op    = (0.4 + Math.random() * 0.4).toFixed(2);
    html += `
      <g class="ws" style="animation-delay:${delay}s;animation-duration:${dur}s">
        <line x1="${x}" y1="160" x2="${x}" y2="${160 - h}"
              stroke="#4a7a30" stroke-width="${(0.8 + Math.random() * 0.6).toFixed(1)}"
              opacity="${op}"/>
        <ellipse cx="${x}" cy="${160 - h}" rx="3" ry="7"
                 fill="#6a9a40" opacity="0.7"
                 transform="rotate(${rot} ${x} ${160 - h})"/>
      </g>`;
  }
  wg.innerHTML = html;
}

/* ── Fireflies ───────────────────────────────────────────────────────────── */
function buildFireflies() {
  const container = document.getElementById("fireflies");
  if (!container) return;
  for (let i = 0; i < 18; i++) {
    const el = document.createElement("div");
    el.className = "firefly";
    el.style.cssText = `
      left:${10 + Math.random() * 80}%;
      top:${40 + Math.random() * 50}%;
      --d:${(6 + Math.random() * 8).toFixed(1)}s;
      --dl:${(-Math.random() * 8).toFixed(1)}s;
      --tx1:${(-30 + Math.random() * 60).toFixed(0)}px; --ty1:${(-40 + Math.random() * 30).toFixed(0)}px;
      --tx2:${(-50 + Math.random() * 100).toFixed(0)}px; --ty2:${(-80 + Math.random() * 60).toFixed(0)}px;
      --tx3:${(-20 + Math.random() * 40).toFixed(0)}px;  --ty3:${(-120 + Math.random() * 80).toFixed(0)}px;
      --tx4:${(-60 + Math.random() * 120).toFixed(0)}px; --ty4:${(-150 + Math.random() * 100).toFixed(0)}px;
    `;
    container.appendChild(el);
  }
}

/* ── Init ────────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  buildInputs();
  attachPresets();
  buildWheat();
  buildFireflies();
});
