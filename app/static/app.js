"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const FIELDS = ["emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
  "background_noise_severity", "audio_quality", "speaker_overlap_present", "long_silence_present"];
const SHORT = { emotional_tone: "tone", emotional_intensity: "intensity", background_noise_present: "noise",
  background_noise_type: "noise type", background_noise_severity: "severity", audio_quality: "quality",
  speaker_overlap_present: "overlap", long_silence_present: "silence" };

let selected = null;       // {files: File[], paths: string[]}
let currentBatch = null;
let expanded = new Set();
let pollTimer = null;

async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  if (r.status === 401) { location.href = "/login"; throw new Error("login required"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function fmtBytes(n) {
  if (n < 1024 ** 2) return (n / 1024).toFixed(0) + " KB";
  if (n < 1024 ** 3) return (n / 1024 ** 2).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

// ---------------------------------------------------------------- selection & upload

$("zip-input").addEventListener("change", (e) => {
  const f = e.target.files[0];
  $("folder-input").value = "";
  selected = f ? { files: [f], paths: [f.name] } : null;
  showSelection();
});

$("folder-input").addEventListener("change", (e) => {
  const files = Array.from(e.target.files);
  $("zip-input").value = "";
  selected = files.length ? { files, paths: files.map((f) => f.webkitRelativePath || f.name) } : null;
  showSelection();
});

function showSelection() {
  $("upload-error").hidden = true;
  if (!selected) { $("selection").textContent = "Nothing selected."; $("upload-btn").disabled = true; return; }
  const size = selected.files.reduce((a, f) => a + f.size, 0);
  const csvs = selected.paths.filter((p) => p.toLowerCase().endsWith(".csv")).length;
  const isZip = selected.files.length === 1 && selected.paths[0].toLowerCase().endsWith(".zip");
  $("selection").innerHTML = isZip
    ? `ZIP: <b>${esc(selected.paths[0])}</b> (${fmtBytes(size)})`
    : `Folder: <b>${selected.files.length}</b> files, ${csvs} CSV (${fmtBytes(size)})`;
  $("upload-btn").disabled = false;
}

$("upload-btn").addEventListener("click", () => {
  if (!selected) return;
  const fd = new FormData();
  selected.files.forEach((f, i) => { fd.append("files", f, f.name); fd.append("paths", selected.paths[i]); });
  fd.append("label", $("batch-label").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/batches");
  $("upload-btn").disabled = true;
  $("upload-progress").hidden = false;
  $("upload-error").hidden = true;
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("upload-progress").firstElementChild.style.width = (100 * e.loaded / e.total) + "%";
  };
  xhr.onload = () => {
    $("upload-progress").hidden = true;
    $("upload-progress").firstElementChild.style.width = "0";
    if (xhr.status === 401) { location.href = "/login"; return; }
    if (xhr.status >= 400) {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail; } catch (_) {}
      $("upload-error").textContent = "Upload failed: " + msg;
      $("upload-error").hidden = false;
      $("upload-btn").disabled = false;
      return;
    }
    const batch = JSON.parse(xhr.responseText);
    selected = null; $("zip-input").value = ""; $("folder-input").value = ""; $("batch-label").value = "";
    showSelection();
    openBatch(batch.id);
    refreshList();
  };
  xhr.onerror = () => {
    $("upload-progress").hidden = true;
    $("upload-error").textContent = "Upload failed: network error.";
    $("upload-error").hidden = false;
    $("upload-btn").disabled = false;
  };
  xhr.send(fd);
});

// ---------------------------------------------------------------- batch list

async function refreshList() {
  const list = await api("/api/batches");
  const tb = $("batch-list");
  if (!list.length) { tb.innerHTML = `<tr><td colspan="5" class="muted">No batches yet.</td></tr>`; return; }
  tb.innerHTML = list.map((b) => {
    const pct = b.total ? Math.round(100 * b.finished / b.total) : 0;
    const errs = (b.counts.error || 0) + (b.counts.missing || 0);
    return `<tr data-id="${esc(b.id)}" class="${b.id === currentBatch ? "active" : ""}">
      <td>${esc(b.created.replace("T", " ").replace("Z", ""))}</td>
      <td>${esc(b.label || b.id)}</td>
      <td><span class="status ${esc(b.status)}">${esc(b.status)}</span></td>
      <td>${b.finished}/${b.total} (${pct}%)</td>
      <td>${errs ? `${errs} file(s)` : ""}${b.n_errors ? ` · ${b.n_errors} batch issue(s)` : ""}</td></tr>`;
  }).join("");
  tb.querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => openBatch(tr.dataset.id)));
}

// ---------------------------------------------------------------- batch detail

function openBatch(id) {
  currentBatch = id;
  expanded = new Set();
  $("detail").hidden = false;
  loadBatch();
  $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadBatch() {
  if (!currentBatch) return;
  let b;
  try { b = await api(`/api/batches/${encodeURIComponent(currentBatch)}`); }
  catch (e) { $("detail").hidden = true; currentBatch = null; return; }
  renderBatch(b);
  clearTimeout(pollTimer);
  if (["queued", "processing", "validating"].includes(b.status)) pollTimer = setTimeout(() => { loadBatch(); refreshList(); }, 2000);
}

function renderBatch(b) {
  const s = b.summary;
  $("detail-title").textContent = b.label ? `${b.label} · ${b.id}` : b.id;
  $("dl-csv").href = `/api/batches/${encodeURIComponent(b.id)}/results.csv`;
  $("dl-json").href = `/api/batches/${encodeURIComponent(b.id)}/results.json`;
  const pct = s.total ? Math.round(100 * s.finished / s.total) : 0;
  $("detail-progress").firstElementChild.style.width = (b.status === "done" ? 100 : pct) + "%";
  const done = s.counts.done || 0, err = s.counts.error || 0, missing = s.counts.missing || 0;
  let line = `<span class="status ${esc(b.status)}">${esc(b.status)}</span> &nbsp;
    ${s.finished} of ${s.total} files finished · ${done} analyzed · ${err} failed`;
  if (missing) line += ` · ${missing} missing from upload`;
  if (b.status === "queued" && b.queue_position) line += ` · waiting (position ${b.queue_position} in queue)`;
  if (b.manifest) line += ` · manifest: <code>${esc(b.manifest)}</code>`;
  $("detail-status").innerHTML = line;

  const order = { error: 0, warning: 1, info: 2 };
  const issues = [...b.issues].sort((a, c) => order[a.level] - order[c.level]);
  $("issues").innerHTML = issues.length
    ? `<h3>Validation report</h3>` + issues.map((i) => `<div class="alert ${esc(i.level)}">${esc(i.level.toUpperCase())}: ${esc(i.message)}</div>`).join("")
    : `<div class="alert info">Manifest and audio files matched with no problems.</div>`;

  const c = b.comparison;
  $("comparison").innerHTML = c
    ? `<h3>Agreement with manifest labels (${c.n} labelled file${c.n === 1 ? "" : "s"})</h3><div class="cmp">` +
      FIELDS.filter((k) => c.fields[k] && c.fields[k].n).map((k) =>
        `<div>${esc(SHORT[k])}<b>${(100 * c.fields[k].accuracy).toFixed(0)}%</b>${c.fields[k].correct}/${c.fields[k].n}</div>`).join("") +
      (c.emotional_tone_macro_f1 != null ? `<div>tone macro-F1<b>${c.emotional_tone_macro_f1.toFixed(2)}</b>&nbsp;</div>` : "") + `</div>`
    : "";

  $("results").innerHTML = b.files.map((f, i) => rowHtml(f, i)).join("");
  $("results").querySelectorAll("tr.row").forEach((tr) => tr.addEventListener("click", () => {
    const i = Number(tr.dataset.i);
    expanded.has(i) ? expanded.delete(i) : expanded.add(i);
    renderBatch(b);
  }));
}

function cell(f, key, text) {
  if (!f.field_match || !(key in f.field_match)) return `<td>${esc(text)}</td>`;
  const ok = f.field_match[key];
  const title = ok ? "matches label" : `label: ${f.label[key] === "" ? '""' : f.label[key]}`;
  return `<td class="${ok ? "match" : "mismatch"}" title="${esc(title)}">${esc(text)}</td>`;
}

function rowHtml(f, i) {
  const r = f.result;
  const yn = (v) => (v ? "yes" : "no");
  let cells;
  if (r) {
    cells = cell(f, "emotional_tone", r.emotional_tone) + cell(f, "emotional_intensity", r.emotional_intensity) +
      cell(f, "background_noise_present", yn(r.background_noise_present)) +
      cell(f, "background_noise_type", r.background_noise_type || "—") +
      cell(f, "background_noise_severity", r.background_noise_severity) + cell(f, "audio_quality", r.audio_quality) +
      cell(f, "speaker_overlap_present", yn(r.speaker_overlap_present)) +
      cell(f, "long_silence_present", yn(r.long_silence_present)) + `<td>${r.confidence.toFixed(2)}</td>`;
  } else {
    cells = `<td colspan="9" class="muted">${esc(f.error || (f.status === "processing" ? "analyzing…" : "waiting"))}</td>`;
  }
  const audio = f.audio_seconds != null ? `${f.audio_seconds.toFixed(1)}s` : "";
  const time = f.processing_seconds != null ? `${f.processing_seconds.toFixed(1)}s` : "";
  let html = `<tr class="row" data-i="${i}"><td>${esc(f.name)}${f.in_manifest === false ? ' <span class="muted small">(not in manifest)</span>' : ""}</td>
    <td><span class="status ${esc(f.status)}">${esc(f.status)}</span></td>${cells}<td>${audio}</td><td>${time}</td></tr>`;
  if (expanded.has(i)) {
    const left = r ? `<b>Output JSON</b><pre>${esc(JSON.stringify(r, null, 2))}</pre>` : `<b>Error</b><pre>${esc(f.error || "not processed yet")}</pre>`;
    const reasons = (f.reasons || []).map((x) => `• ${esc(x)}`).join("\n");
    const label = f.label ? `<b>Label from manifest</b><pre>${esc(JSON.stringify(f.label, null, 2))}</pre>` : "";
    html += `<tr class="expand"><td colspan="13"><div class="expand-grid"><div>${left}${label}</div>
      <div><b>Why</b><pre>${reasons || "—"}</pre>${f.language ? `<p class="muted small">Detected language: ${esc(f.language)}</p>` : ""}</div></div></td></tr>`;
  }
  return html;
}

$("delete-btn").addEventListener("click", async () => {
  if (!currentBatch || !confirm("Delete this batch and its results?")) return;
  try { await api(`/api/batches/${encodeURIComponent(currentBatch)}`, { method: "DELETE" }); }
  catch (e) { alert(e.message); return; }
  currentBatch = null; $("detail").hidden = true; refreshList();
});

// ---------------------------------------------------------------- model status

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const el = $("model-status");
    if (s.warmup_error) { el.className = "pill error"; el.textContent = "model load failed"; el.title = s.warmup_error; }
    else if (s.models_ready) { el.className = "pill ready"; el.textContent = "models ready"; }
    else { el.className = "pill"; el.textContent = "loading models… (first start takes ~1 min)"; setTimeout(refreshStatus, 3000); }
  } catch (_) { setTimeout(refreshStatus, 5000); }
}

refreshStatus();
refreshList();
setInterval(refreshList, 10000);
