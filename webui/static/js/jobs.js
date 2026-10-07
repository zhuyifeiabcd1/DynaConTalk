// Jobs drawer: list, filters, star / cancel / delete, logs, job selection + live status (SSE).
import { $, $$, el, icon, toast, fmtDuration, relTime, confirm, debounce } from "./util.js";
import { t, onLang } from "./i18n.js";
import { api } from "./api.js";
import { state, on, emit, set, loadPending, seek } from "./store.js";
import { shortId } from "./stage.js";

const ACTIVE = new Set(["queued", "waiting", "running", "starting", "cancelling"]);
let filter = "all";
let query = "";
let source = null;          // EventSource for the selected job
let logOffset = 0;
let logTimer = null;
let drawerTab = "jobs";

export function initJobs() {
  $$("#drawerTabs button").forEach((b) => b.addEventListener("click", () => setDrawerTab(b.dataset.tab)));
  $("#jobsFilter").addEventListener("change", (e) => { filter = e.target.value; renderList(); });
  $("#jobsSearch").addEventListener("input", debounce((e) => { query = e.target.value.trim().toLowerCase(); renderList(); }, 120));
  $("#drawerToggle").addEventListener("click", () => $("#drawer").classList.toggle("collapsed"));
  $("#logFollow").addEventListener("change", () => { if ($("#logFollow").checked) scrollLog(); });
  on("jobs", renderList);
  on("job", () => { renderList(); restartLogs(); });
  on("jobDeleted", (id) => { if (state.job && state.job.job_id === id) { closeSource(); state.job = null; emit("job", state); } refreshJobs(); });
  on("openJob", (id) => selectJob(id));
  onLang(renderList);
  setInterval(pollVersion, 2500);
  setInterval(refreshSystem, 10000);
  refreshSystem();
}

function setDrawerTab(tab) {
  drawerTab = tab;
  $$("#drawerTabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $("#jobsList").classList.toggle("hidden", tab !== "jobs");
  $("#logWrap").classList.toggle("hidden", tab !== "logs");
  $("#drawer").classList.remove("collapsed");
  if (tab === "logs") restartLogs();
}

export async function refreshJobs() {
  try {
    const data = await api.jobs();
    state.jobs = data.items || [];
    state.jobsVersion = data.version;
    // keep the selected job's summary fields fresh without a full detail fetch
    if (state.job) {
      const s = state.jobs.find((j) => j.job_id === state.job.job_id);
      if (s) { Object.assign(state.job, { state: s.state, phase: s.phase, progress: s.progress, message: s.message, elapsed_seconds: s.elapsed_seconds, title: s.title, starred: s.starred }); emit("jobStatus", state); }
    }
    emit("jobs", state);
    transcribeWatch();
  } catch (err) { console.warn("jobs", err); }
}

async function pollVersion() {
  try {
    const v = await api.jobsVersion();
    if (v.version !== state.jobsVersion) await refreshJobs();
    else if (v.active && (state.jobs.some((j) => ACTIVE.has(j.state)))) await refreshJobs();
  } catch (_) {}
}

export async function refreshSystem() {
  try { state.system = await api.system(); emit("system", state); } catch (_) {}
}

export async function selectJob(id, { keepPlayhead = false } = {}) {
  if (!id) return;
  try {
    const detail = await api.job(id);
    const changed = !state.job || state.job.job_id !== id;
    state.job = detail;
    state.mode = "job";
    if (changed) { state.selection = null; state.ab = "edited"; $$("#abToggle button").forEach((b) => b.classList.toggle("on", b.dataset.ab === "edited")); }
    loadPending();
    emit("job", state);
    if (!keepPlayhead || changed) seek(changed ? 0 : state.playhead, "ui");
    watch(detail);
    $("#jobTitle").value = detail.title || "";
    await loadWords(detail);
  } catch (err) { toast(err.message, "err"); }
}

async function loadWords(detail) {
  const key = detail.root_job_id || detail.job_id;
  if (!detail.words_available) { if (state.words) { state.words = null; emit("words", state); } else emit("words", state); return; }
  if (state.words && state.words.job_id === key) return;
  try { state.words = await api.words(detail.job_id); } catch (_) { state.words = null; }
  emit("words", state);
}

let lastTranscribeDone = new Set();
function transcribeWatch() {
  const done = new Set(state.jobs.filter((j) => j.action === "transcribe" && j.state === "succeeded").map((j) => j.job_id));
  const fresh = [...done].some((id) => !lastTranscribeDone.has(id));
  lastTranscribeDone = done;
  if (fresh && state.job) refreshJob();
}

export async function refreshJob() {
  if (!state.job) return;
  await selectJob(state.job.job_id, { keepPlayhead: true });
}

function closeSource() { if (source) { source.close(); source = null; } }

function watch(job) {
  closeSource();
  if (!ACTIVE.has(job.state)) return;
  source = api.events(job.job_id);
  source.addEventListener("status", (e) => {
    try {
      const payload = JSON.parse(e.data);
      const st = payload.data && payload.data.status;
      if (st && state.job && state.job.job_id === st.job_id) {
        Object.assign(state.job, { state: st.state, phase: st.phase, progress: st.progress, message: st.message, status: st });
        emit("jobStatus", state);
        renderList();
      }
    } catch (_) {}
  });
  source.addEventListener("done", async () => { closeSource(); await refreshJobs(); await refreshJob(); const j = state.job; if (j) toast(`${shortId(j.job_id)} · ${j.state}`, j.state === "succeeded" ? "ok" : "err"); });
  source.onerror = () => { /* the poller keeps things fresh */ };
}

// ------------------------------------------------------------------ list
function visibleJobs() {
  let list = state.jobs;
  if (filter === "active") list = list.filter((j) => ACTIVE.has(j.state));
  else if (filter === "failed") list = list.filter((j) => j.state === "failed" || j.state === "cancelled");
  else if (filter === "starred") list = list.filter((j) => j.starred);
  else if (filter === "generate" || filter === "edit") list = list.filter((j) => j.action === filter);
  if (query) list = list.filter((j) => [j.job_id, j.title, j.audio_name, j.identity, j.trajectory, j.notes].join(" ").toLowerCase().includes(query));
  return list;
}

function renderList() {
  const host = $("#jobsList");
  const keepScroll = host.scrollTop;
  host.innerHTML = "";
  const list = visibleJobs();
  $("#jobsCount").textContent = String(state.jobs.length);
  if (!list.length) { host.append(el("div", { class: "empty", style: { gridColumn: "1 / -1" } }, [icon("layers"), el("div", { text: t("library_empty") })])); return; }
  const cur = state.job && state.job.job_id;
  list.forEach((j) => host.append(jobRow(j, j.job_id === cur)));
  host.scrollTop = keepScroll;
}

function jobRow(j, on) {
  const active = ACTIVE.has(j.state);
  const row = el("div", { class: `job-row ${on ? "on" : ""}`, dataset: { id: j.job_id }, onclick: () => selectJob(j.job_id) });
  const star = el("button", { class: `star ${j.starred ? "on" : ""}`, title: t("starred"), onclick: async (e) => { e.stopPropagation(); try { await api.patchJob(j.job_id, { starred: !j.starred }); j.starred = !j.starred; star.classList.toggle("on", j.starred); } catch (err) { toast(err.message, "err"); } } }, [icon("star")]);
  const main = el("div", { class: "main" }, [
    el("div", { class: "t1" }, [el("span", { class: `badge ${j.action}`, text: j.action }), el("span", { class: "name", text: j.title || j.audio_name || shortId(j.job_id) })]),
    el("div", { class: "t2" }, [
      el("span", { class: "mono", text: shortId(j.job_id) }),
      j.identity ? el("span", { text: j.identity }) : null,
      j.keypose_count ? el("span", { text: `${j.keypose_count} kp` }) : null,
      j.traj_count ? el("span", { text: `${j.traj_count} mv` }) : null,
      el("span", { text: active ? (j.message || "") : relTime(j.created_at, t) }),
    ]),
    active ? el("div", { class: "prog" }, [el("i", { style: { width: `${Math.round((j.progress || 0) * 100)}%` } })]) : null,
  ]);
  const right = el("div", { class: "right" }, [
    el("span", { class: `pill pill-${j.state}`, text: j.state }),
    el("div", { class: "acts" }, [
      active ? el("button", { title: t("cancel_job"), onclick: async (e) => { e.stopPropagation(); try { await api.cancelJob(j.job_id); toast(t("cancelled"), "warn"); refreshJobs(); } catch (err) { toast(err.message, "err"); } } }, [icon("cancel")]) : null,
      !active ? el("button", { class: "del", title: t("delete"), onclick: async (e) => { e.stopPropagation(); if (await confirm(`${t("confirm_delete")}\n${j.job_id}`, { ok: t("delete"), cancel: t("cancel"), danger: true })) { try { await api.deleteJob(j.job_id); toast(t("deleted"), "ok"); emit("jobDeleted", j.job_id); } catch (err) { toast(err.message, "err"); } } } }, [icon("trash")]) : null,
    ]),
    j.elapsed_seconds ? el("span", { class: "muted mono", style: { fontSize: "11px" }, text: fmtDuration(j.elapsed_seconds) }) : null,
  ]);
  row.append(star, main, right);
  return row;
}

// ------------------------------------------------------------------ logs
function restartLogs() {
  clearInterval(logTimer);
  logTimer = null;
  logOffset = 0;
  $("#logView").textContent = "";
  $("#logMeta").textContent = state.job ? state.job.job_id : "";
  if (!state.job || drawerTab !== "logs") return;
  fetchLogs();
  logTimer = setInterval(() => { if (!state.job) return; if (ACTIVE.has(state.job.state)) fetchLogs(); }, 2000);
}

async function fetchLogs() {
  if (!state.job) return;
  try {
    const data = await api.logs(state.job.job_id, logOffset);
    if (data.text) {
      const view = $("#logView");
      if (logOffset === 0) view.textContent = data.text; else view.textContent += data.text;
      logOffset = data.offset;
      scrollLog();
    }
    $("#logMeta").textContent = `${state.job.job_id} · ${Math.round((data.size || 0) / 1024)} KB`;
  } catch (_) {}
}

function scrollLog() {
  if ($("#logFollow").checked) { const v = $("#logView"); v.scrollTop = v.scrollHeight; }
}
