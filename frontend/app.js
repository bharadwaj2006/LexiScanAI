/* ─────────────────────────────────────────────────────────────────
   LexiScan Auto — Frontend Logic
   ───────────────────────────────────────────────────────────────── */

const API_BASE = "";          // same origin when served by FastAPI
let currentFile = null;
let lastJSON    = null;

/* ── Tab switching ──────────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("tab-" + tab).classList.add("active");
  document.getElementById("panel-" + tab).classList.add("active");
}

/* ── File handling ──────────────────────────────────────────────── */
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) attachFile(file);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) attachFile(file);
}

function attachFile(file) {
  currentFile = file;
  const allowed = ["application/pdf", "text/plain"];
  if (!allowed.includes(file.type) && !file.name.endsWith(".pdf") && !file.name.endsWith(".txt")) {
    showToast("Only PDF and TXT files are supported.", "error");
    return;
  }
  document.getElementById("filePreview").hidden = false;
  document.getElementById("dropZone").querySelector(".drop-icon").style.display = "none";
  document.getElementById("dropZone").querySelector(".drop-text").style.display = "none";
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("fileSize").textContent = formatBytes(file.size);
  document.getElementById("extractBtnFile").disabled = false;
}

function clearFile() {
  currentFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("filePreview").hidden = true;
  document.getElementById("dropZone").querySelector(".drop-icon").style.display = "";
  document.getElementById("dropZone").querySelector(".drop-text").style.display = "";
  document.getElementById("extractBtnFile").disabled = true;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

/* ── Extraction ─────────────────────────────────────────────────── */
async function extractFromFile() {
  if (!currentFile) return;
  const formData = new FormData();
  formData.append("file", currentFile);
  await runExtraction(formData);
}

async function extractFromText() {
  const text = document.getElementById("textInput").value.trim();
  if (!text) { showToast("Please paste some contract text first.", "error"); return; }
  const formData = new FormData();
  formData.append("text", text);
  await runExtraction(formData);
}

async function runDemo() {
  showProgress();
  animatePipeline();
  try {
    const res = await authFetch(`${API_BASE}/api/demo`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    finishProgress();
    setTimeout(() => renderResults(data), 400);
  } catch (err) {
    hideProgress();
    showToast("Demo request failed: " + err.message, "error");
  }
}

async function runExtraction(formData) {
  showProgress();
  animatePipeline();
  try {
    const res = await authFetch(`${API_BASE}/api/extract`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    finishProgress();
    setTimeout(() => renderResults(data), 400);
  } catch (err) {
    hideProgress();
    showToast("Extraction failed: " + err.message, "error");
  }
}

/* ── Progress pipeline ──────────────────────────────────────────── */
const STEPS = ["step-upload", "step-ocr", "step-ner", "step-validate", "step-done"];
const LABELS = [
  "Uploading document…",
  "Running OCR pipeline (Tesseract)…",
  "Running NER model (SpaCy)…",
  "Validating & normalising entities…",
  "Complete!",
];
let stepTimer = null;

function showProgress() {
  document.getElementById("progressWrap").hidden = false;
  document.getElementById("results").hidden = true;
  STEPS.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove("active", "done");
  });
  setProgress(0, "Initialising pipeline…");
}

function hideProgress() {
  document.getElementById("progressWrap").hidden = true;
}

function animatePipeline() {
  let idx = 0;
  clearInterval(stepTimer);
  const advance = () => {
    if (idx >= STEPS.length) { clearInterval(stepTimer); return; }
    // Mark previous as done
    if (idx > 0) {
      document.getElementById(STEPS[idx - 1]).classList.remove("active");
      document.getElementById(STEPS[idx - 1]).classList.add("done");
    }
    document.getElementById(STEPS[idx]).classList.add("active");
    setProgress(Math.round((idx / (STEPS.length - 1)) * 85), LABELS[idx]);
    idx++;
  };
  advance();
  stepTimer = setInterval(advance, 900);
}

function finishProgress() {
  clearInterval(stepTimer);
  STEPS.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove("active");
    el.classList.add("done");
  });
  setProgress(100, "Complete!");
}

function setProgress(pct, label) {
  document.getElementById("progressFill").style.width = pct + "%";
  document.getElementById("progressLabel").textContent = label;
}

/* ── Results rendering ──────────────────────────────────────────── */
function renderResults(data) {
  hideProgress();
  lastJSON = data;

  const entities = data.entities || {};
  const dates    = entities.dates    || [];
  const parties  = entities.parties  || [];
  const amounts  = entities.amounts  || [];
  const terms    = entities.termination_clauses || [];

  // Summary bar
  const ocr = data.ocr_used ? `<span class="meta-tag">OCR ✓</span>` : `<span class="meta-tag">Native PDF</span>`;
  document.getElementById("summaryMeta").innerHTML =
    `Processed in <strong>${data.processing_time_ms}ms</strong> · ${data.page_count} page(s) · ${data.character_count?.toLocaleString()} chars · ${ocr}`;

  document.getElementById("summaryCounts").innerHTML = [
    badge(dates.length,   "Dates",    "#06b6d4"),
    badge(parties.length, "Parties",  "#a855f7"),
    badge(amounts.length, "Amounts",  "#10b981"),
    badge(terms.length,   "Clauses",  "#f59e0b"),
  ].join("");

  // Populate entity lists
  renderDates(dates);
  renderParties(parties);
  renderAmounts(amounts);
  renderTerminations(terms);

  // JSON viewer
  document.getElementById("jsonBody").textContent =
    JSON.stringify(data, null, 2);

  // Show results
  document.getElementById("results").hidden = false;
  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function badge(count, label, color) {
  return `<span class="count-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">${count} ${label}</span>`;
}

function renderDates(dates) {
  const ul = document.getElementById("list-dates");
  document.getElementById("count-dates").textContent = `${dates.length} found`;
  if (!dates.length) { ul.innerHTML = `<li class="empty-state">No dates detected</li>`; return; }
  ul.innerHTML = dates.map((d, i) => `
    <li class="entity-item" style="animation-delay:${i * 0.06}s">
      <div class="entity-text">${escHtml(d.text)}</div>
      <div class="entity-meta">
        ${d.normalized ? `<span class="meta-tag">📅 ${d.normalized}</span>` : ""}
        <span class="meta-tag">${Math.round((d.confidence || 0.9) * 100)}% conf</span>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${Math.round((d.confidence || 0.9) * 100)}%;background:linear-gradient(90deg,#06b6d4,#a855f7)"></div>
      </div>
    </li>`).join("");
}

function renderParties(parties) {
  const ul = document.getElementById("list-parties");
  document.getElementById("count-parties").textContent = `${parties.length} found`;
  if (!parties.length) { ul.innerHTML = `<li class="empty-state">No parties detected</li>`; return; }
  ul.innerHTML = parties.map((p, i) => `
    <li class="entity-item" style="animation-delay:${i * 0.06}s">
      <div class="entity-text">${escHtml(p.text)}</div>
      <div class="entity-meta">
        <span class="meta-tag">PARTY</span>
        <span class="meta-tag">${Math.round((p.confidence || 0.87) * 100)}% conf</span>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${Math.round((p.confidence || 0.87) * 100)}%;background:linear-gradient(90deg,#a855f7,#06b6d4)"></div>
      </div>
    </li>`).join("");
}

function renderAmounts(amounts) {
  const ul = document.getElementById("list-amounts");
  document.getElementById("count-amounts").textContent = `${amounts.length} found`;
  if (!amounts.length) { ul.innerHTML = `<li class="empty-state">No amounts detected</li>`; return; }
  ul.innerHTML = amounts.map((a, i) => `
    <li class="entity-item" style="animation-delay:${i * 0.06}s">
      <div class="entity-text">${escHtml(a.text)}</div>
      <div class="entity-meta">
        ${a.normalized ? `<span class="meta-tag">${escHtml(a.normalized)}</span>` : ""}
        ${a.currency ? `<span class="meta-tag">${a.currency}</span>` : ""}
        <span class="meta-tag">${Math.round((a.confidence || 0.94) * 100)}% conf</span>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${Math.round((a.confidence || 0.94) * 100)}%;background:linear-gradient(90deg,#10b981,#06b6d4)"></div>
      </div>
    </li>`).join("");
}

function renderTerminations(terms) {
  const ul = document.getElementById("list-term");
  document.getElementById("count-term").textContent = `${terms.length} found`;
  if (!terms.length) { ul.innerHTML = `<li class="empty-state">No termination clauses detected</li>`; return; }
  ul.innerHTML = terms.map((t, i) => `
    <li style="animation-delay:${i * 0.06}s">
      <div class="entity-clause">${escHtml(t.text)}</div>
      <div class="entity-meta" style="margin-top:6px">
        <span class="meta-tag">TERMINATION_CLAUSE</span>
        <span class="meta-tag">${Math.round((t.confidence || 0.88) * 100)}% conf</span>
      </div>
    </li>`).join("");
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Copy JSON ──────────────────────────────────────────────────── */
function copyJSON() {
  if (!lastJSON) return;
  navigator.clipboard.writeText(JSON.stringify(lastJSON, null, 2)).then(() => {
    const btn = document.getElementById("copyBtn");
    btn.textContent = "✓ Copied!";
    setTimeout(() => { btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`; }, 2000);
  });
}

/* ── Reset ──────────────────────────────────────────────────────── */
function resetUI() {
  document.getElementById("results").hidden = true;
  clearFile();
  document.getElementById("textInput").value = "";
  lastJSON = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ── Toast notification ─────────────────────────────────────────── */
function showToast(msg, type = "info") {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    padding:12px 20px; border-radius:10px; font-size:0.875rem;
    background:${type === "error" ? "#ef444422" : "#10b98122"};
    border:1px solid ${type === "error" ? "#ef4444" : "#10b981"};
    color:${type === "error" ? "#ef4444" : "#10b981"};
    animation:fadeUp 0.3s ease; max-width:340px;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
