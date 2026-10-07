// Entry point: boot, topbar, global shortcuts, drag & drop, system chip.
import { $, $$, el, icon, toast, modal, isTyping, confirm } from "./util.js";
import { t, initLang, setLang, getLang, onLang } from "./i18n.js";
import { api } from "./api.js";
import { state, on, emit, seek, timelineSource, chunkAt, clearPending, pendingCount } from "./store.js";
import { initLibrary, loadAssets, setTab as setLibraryTab } from "./library.js";
import { initTimeline, setZoom, removeSelected } from "./timeline.js";
import { initStage, togglePlay, step, insertKeyposeAtPlayhead, addMoveAtPlayhead } from "./stage.js";
import { initInspector, showTab, applyEdits, setAudioFile } from "./inspector.js";
import { initJobs, refreshJobs, selectJob } from "./jobs.js";

// runtime error collector (visible to headless smoke tests via #errlog)
window.__errors = [];
const errlog = document.createElement("div"); errlog.id = "errlog"; errlog.hidden = true; document.body.append(errlog);
const record = (m) => { window.__errors.push(m); errlog.textContent = window.__errors.join("\n"); };
window.addEventListener("error", (e) => record(`${e.message} @ ${e.filename}:${e.lineno}`));
window.addEventListener("unhandledrejection", (e) => record(`unhandled: ${e.reason && (e.reason.stack || e.reason.message || e.reason)}`));

async function boot() {
  initLang();
  $$("#langToggle button").forEach((b) => { b.classList.toggle("on", b.dataset.lang === getLang()); b.addEventListener("click", () => { setLang(b.dataset.lang); $$("#langToggle button").forEach((x) => x.classList.toggle("on", x === b)); }); });
  initLibrary();
  initTimeline();
  initStage();
  initInspector();
  initJobs();
  initTopbar();
  initShortcuts();
  initGlobalDrop();
  if (window.innerHeight < 860) $("#drawer").classList.add("collapsed");
  $("#btnLibrary").addEventListener("click", (e) => { const off = $("#app").classList.toggle("lib-collapsed"); e.currentTarget.classList.toggle("on", !off); });
  on("system", renderGpuChip);
  on("libraryTab", (tab) => setLibraryTab(tab));
  await Promise.all([loadAssets(), refreshJobs()]);
  // restore the last opened job
  const last = (() => { try { return localStorage.getItem("studio.lastJob"); } catch (_) { return null; } })();
  const fromHash = location.hash.replace(/^#job=/, "");
  const target = fromHash || last;
  if (target && state.jobs.some((j) => j.job_id === target)) await selectJob(target);
  else if (!state.jobs.length) showTab("generate");
  on("job", () => { try { if (state.job) { localStorage.setItem("studio.lastJob", state.job.job_id); if (location.hash !== `#job=${state.job.job_id}`) history.replaceState(null, "", `#job=${state.job.job_id}`); } } catch (_) {} renderTitle(); });
  // deep links / back-forward: #job=<id>
  window.addEventListener("hashchange", () => {
    const id = location.hash.replace(/^#job=/, "");
    if (id && (!state.job || state.job.job_id !== id)) selectJob(id);
  });
  window.__studio = { state };
  on("jobStatus", renderTitle);
  onLang(renderTitle);
  renderTitle();
}

function initTopbar() {
  $("#btnNew").addEventListener("click", () => { showTab("generate"); $("#audioInput").click(); });
  $("#btnHelp").addEventListener("click", showHelp);
  const title = $("#jobTitle");
  title.addEventListener("change", async () => {
    if (!state.job) return;
    try { await api.patchJob(state.job.job_id, { title: title.value }); state.job.title = title.value; emit("jobStatus", state); await refreshJobs(); } catch (err) { toast(err.message, "err"); }
  });
  title.addEventListener("keydown", (e) => { if (e.key === "Enter") title.blur(); });
  $("#btnApply").addEventListener("click", applyEdits);
  $("#btnDiscard").addEventListener("click", async () => { if (await confirm(t("confirm_discard"), { ok: t("discard"), cancel: t("cancel"), danger: true })) clearPending(); });
}

function renderTitle() {
  const job = state.job;
  const pill = $("#jobStatePill");
  const title = $("#jobTitle");
  if (state.mode === "draft" && state.draft) {
    pill.className = "pill pill-draft"; pill.textContent = t("draft");
    title.value = state.draft.name || ""; title.disabled = true;
    $("#jobPhase").textContent = "";
    return;
  }
  title.disabled = !job;
  if (!job) { pill.className = "pill pill-idle"; pill.textContent = t("idle"); title.value = ""; $("#jobPhase").textContent = ""; return; }
  pill.className = `pill pill-${job.state}`; pill.textContent = job.state;
  title.placeholder = job.audio_name || t("untitled_job");
  if (document.activeElement !== title) title.value = job.title || "";
  const active = ["queued", "waiting", "running", "starting", "cancelling"].includes(job.state);
  $("#jobPhase").textContent = active ? `${job.phase || ""} ${Math.round((job.progress || 0) * 100)}% · ${job.message || ""}` : "";
}

function renderGpuChip() {
  const sys = state.system;
  const chip = $("#gpuChip"), text = $("#gpuText");
  if (!sys || !sys.gpus || !sys.gpus.length) { chip.className = "chip gpu-chip off"; text.textContent = "GPU ?"; return; }
  const g = sys.gpus.find((x) => String(x.index) === String(sys.gpu_index)) || sys.gpus[0];
  const free = g.memory_total_mb - g.memory_used_mb;
  const busy = free < (sys.min_free_gpu_mb || 0);
  chip.className = `chip gpu-chip ${busy ? "busy" : "free"}`;
  const q = (sys.queue || []).length;
  text.textContent = `GPU${g.index} ${(g.memory_used_mb / 1024).toFixed(1)}/${(g.memory_total_mb / 1024).toFixed(0)} GB · ${g.utilization}%${q ? ` · ${q} queued` : ""}`;
  chip.title = `${g.name}\n${busy ? t("gpu_busy") : t("gpu_free")}${sys.current_job ? `\n${t("job")}: ${sys.current_job}` : ""}${sys.disk && sys.disk.free_gb ? `\n${t("disk_free")}: ${sys.disk.free_gb} GB` : ""}`;
}

function initShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (isTyping()) return;
    const src = timelineSource();
    const mult = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case " ": e.preventDefault(); togglePlay(); break;
      case "ArrowLeft": e.preventDefault(); step(-mult); break;
      case "ArrowRight": e.preventDefault(); step(mult); break;
      case "Home": e.preventDefault(); seek(0, "ui"); break;
      case "End": e.preventDefault(); seek(src.totalFrames - 1, "ui"); break;
      case "k": case "K": insertKeyposeAtPlayhead(); break;
      case "t": case "T": addMoveAtPlayhead(); break;
      case "Delete": case "Backspace": if (removeSelected()) e.preventDefault(); break;
      case "Escape": state.selection = null; emit("selection", state); break;
      case "f": case "F": setZoom(0); break;
      case "[": { const c = chunkAt(state.playhead); const i = src.chunks.indexOf(c); if (i > 0) seek(src.chunks[i - 1].frame_start, "ui"); break; }
      case "]": { const c = chunkAt(state.playhead); const i = src.chunks.indexOf(c); if (i >= 0 && i < src.chunks.length - 1) seek(src.chunks[i + 1].frame_start, "ui"); break; }
      case "Enter": if (e.ctrlKey || e.metaKey) { e.preventDefault(); applyEdits(); } break;
      case "?": showHelp(); break;
      default: return;
    }
  });
}

function showHelp() {
  modal((card, close) => {
    const rows = [
      ["Space", t("help_space")], ["← / →", t("help_arrows")], ["Home / End", t("help_home")], ["K", t("help_k")], ["T", t("help_t")],
      ["Delete", t("help_del")], ["Esc", t("help_esc")], ["Ctrl + Enter", t("help_apply")], ["[ / ]", t("help_brackets")], ["F", t("help_f")], ["Ctrl + wheel", "Zoom timeline"],
    ];
    card.append(
      el("h2", { text: t("shortcuts") }),
      el("div", { class: "shortcuts" }, rows.map(([k, v]) => el("div", {}, [el("span", { text: v }), el("kbd", { text: k })]))),
      el("p", { class: "hint", style: { marginTop: "14px" }, text: t("help_drag") }),
      el("div", { class: "actions" }, [el("button", { class: "btn primary", onclick: close }, ["OK"])]),
    );
  });
}

function initGlobalDrop() {
  const overlay = $("#dropOverlay");
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; depth += 1; overlay.classList.remove("hidden"); });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) overlay.classList.add("hidden"); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0; overlay.classList.add("hidden");
    const file = [...e.dataTransfer.files].find((f) => /audio|\.(wav|mp3|flac|m4a|ogg)$/i.test(`${f.type} ${f.name}`));
    if (!file) { toast(t("no_audio"), "warn"); return; }
    const dt = new DataTransfer(); dt.items.add(file);
    $("#audioInput").files = dt.files;
    setAudioFile(file);
  });
}

function hasFiles(e) { return e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files"); }

boot().catch((err) => { console.error(err); toast(`${t("server_error")}: ${err.message}`, "err", 8000); });
