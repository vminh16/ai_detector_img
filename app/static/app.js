/* app/static/app.js — Frontend logic for AI Image Detector */
"use strict";

const $ = (sel) => document.querySelector(sel);
const API_PREDICT = "/api/predict";

// ── DOM refs ─────────────────────────────────────────────────────
const dropZone       = $("#dropZone");
const fileInput      = $("#fileInput");
const previewSection = $("#previewSection");
const previewImage   = $("#previewImage");
const fileName       = $("#fileName");
const fileSize       = $("#fileSize");
const analyzeBtn     = $("#analyzeBtn");
const changeBtn      = $("#changeBtn");
const btnText        = analyzeBtn.querySelector(".btn-text");
const btnSpinner     = analyzeBtn.querySelector(".btn-spinner");
const placeholderSec = $("#placeholderSection");
const resultSection  = $("#resultSection");
const errorSection   = $("#errorSection");
const errorMessage   = $("#errorMessage");
const resetBtn       = $("#resetBtn");
const errorResetBtn  = $("#errorResetBtn");

let currentFile = null;
let clientHash  = null;

// ── File selection ──────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported format. Please use JPEG, PNG, or WebP.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("File too large (max 20 MB).");
    return;
  }

  currentFile = file;
  clientHash = null;

  previewImage.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);

  computeSHA256(file).then((h) => { clientHash = h; });

  // Show preview, hide drop zone
  dropZone.classList.add("hidden");
  previewSection.classList.remove("hidden");
  // Keep right panel as-is (placeholder or previous result)
}

// ── Change / remove image ───────────────────────────────────────
changeBtn.addEventListener("click", resetAll);

// ── Analyze ─────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", () => {
  if (!currentFile) return;
  runPrediction(currentFile);
});

async function runPrediction(file) {
  setLoading(true);
  // hide right-panel sections, keep preview
  placeholderSec.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch(API_PREDICT, { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render result ───────────────────────────────────────────────
function renderResult(data) {
  placeholderSec.classList.add("hidden");
  resultSection.classList.remove("hidden");

  const score = data.calibrated_score;
  $("#scoreValue").textContent = score.toFixed(4);

  // Gauge
  const arc = $("#gaugeArc");
  const maxLen = 251.33;
  arc.style.strokeDashoffset = maxLen * (1 - score);
  arc.style.stroke = scoreColor(score);
  $("#scoreValue").style.color = scoreColor(score);

  // Elapsed
  $("#elapsedTime").textContent = `${data.elapsed_ms.toFixed(0)} ms`;

  // Verdict
  const zone = data.zone;
  const vc = $("#verdictCard");
  vc.className = "verdict-card zone-" + zone;

  const verdicts = {
    LOW:    { icon: "✅", label: "Likely Real",         desc: "Low probability of AI generation." },
    MEDIUM: { icon: "⚠️", label: "Uncertain — Review",  desc: "Ambiguous zone. Manual review recommended." },
    HIGH:   { icon: "🚨", label: "Likely AI-Generated", desc: "High probability of AI generation." },
  };
  const v = verdicts[zone] || verdicts.MEDIUM;
  $("#verdictIcon").textContent = v.icon;
  $("#verdictLabel").textContent = v.label;
  $("#verdictDesc").textContent = v.desc;

  // Detail chips
  const zv = $("#zoneValue");
  zv.textContent = zone;
  zv.className = "chip-val badge badge-" + zone;
  $("#decisionValue").textContent = data.decision;
  $("#rawScoreValue").textContent = data.raw_score.toFixed(6);
  $("#formatValue").textContent = data.format_detected.toUpperCase();

  // Contributors
  const list = $("#contributorsList");
  list.innerHTML = "";
  (data.top_contributors || []).forEach((c) => {
    const pct = (c.importance * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "contrib-row";
    row.innerHTML = `
      <span class="contrib-name">${escapeHtml(c.feature)}</span>
      <div class="contrib-bar-wrap">
        <div class="contrib-bar" style="width:${pct}%"></div>
      </div>
      <span class="contrib-pct">${pct}%</span>`;
    list.appendChild(row);
  });

  // Integrity
  const ch = clientHash || "computing…";
  const sh = data.server_hash || "—";
  $("#clientHash").textContent = ch;
  $("#serverHash").textContent = sh;
  const match = ch === sh;
  const hm = $("#hashMatch");
  hm.textContent = match ? "✅ Match" : "❌ Mismatch";
  hm.style.color = match ? "var(--green)" : "var(--red)";
}

// ── Reset ───────────────────────────────────────────────────────
resetBtn.addEventListener("click", resetAll);
errorResetBtn.addEventListener("click", resetAll);

function resetAll() {
  currentFile = null;
  clientHash = null;
  fileInput.value = "";
  previewImage.src = "";

  // Left panel: show drop zone, hide preview
  dropZone.classList.remove("hidden");
  previewSection.classList.add("hidden");

  // Right panel: show placeholder, hide results/error
  placeholderSec.classList.remove("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");
}

// ── Helpers ─────────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  btnText.classList.toggle("hidden", on);
  btnSpinner.classList.toggle("hidden", !on);
}

function showError(msg) {
  placeholderSec.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorMessage.textContent = msg;
  errorSection.classList.remove("hidden");
}

function scoreColor(s) {
  if (s < 0.3)  return "var(--green)";
  if (s < 0.787516) return "var(--yellow)";
  return "var(--red)";
}

function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function escapeHtml(t) {
  const d = document.createElement("div");
  d.textContent = t;
  return d.innerHTML;
}

async function computeSHA256(file) {
  const buf = await file.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, "0")).join("");
}
