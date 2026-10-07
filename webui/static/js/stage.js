// Stage: video player synced to the global playhead, A/B compare, transport, path minimap.
import { $, $$, el, icon, fmtTime, clamp, debounce } from "./util.js";
import { t, onLang } from "./i18n.js";
import { api } from "./api.js";
import { state, on, emit, seek, timelineSource, chunkAt, addTraj, addKeypose, sentenceAt, wordAt } from "./store.js";
import { storage } from "./util.js";
import { toast } from "./util.js";

const video = () => $("#video");
let currentUrl = "";
let raf = 0;
let seekingFromUi = false;
let loop = false;
let previewToken = 0;

export function initStage() {
  const v = video();
  v.addEventListener("loadedmetadata", () => syncVideoToPlayhead());
  v.addEventListener("play", () => { state.playing = true; emit("playing", state); tick(); updatePlayButton(); });
  v.addEventListener("pause", () => { state.playing = false; emit("playing", state); cancelAnimationFrame(raf); updatePlayButton(); });
  v.addEventListener("ended", () => onEnded());
  v.addEventListener("error", () => { if (v.getAttribute("src")) console.warn("video error", v.error); $("#videoWrap").classList.remove("loading"); });
  ["loadstart", "waiting", "seeking"].forEach((ev) => v.addEventListener(ev, () => { if (v.getAttribute("src")) $("#videoWrap").classList.add("loading"); }));
  ["canplay", "playing", "seeked", "loadeddata"].forEach((ev) => v.addEventListener(ev, () => $("#videoWrap").classList.remove("loading")));

  $("#btnPlay").addEventListener("click", togglePlay);
  $("#btnStepBack").addEventListener("click", () => step(-1));
  $("#btnStepFwd").addEventListener("click", () => step(1));
  $("#speedSel").addEventListener("change", (e) => { v.playbackRate = Number(e.target.value); });
  $("#btnLoop").addEventListener("click", (e) => { loop = !loop; e.currentTarget.classList.toggle("on", loop); });
  $("#btnFullscreen").addEventListener("click", () => { const w = $("#videoWrap"); if (document.fullscreenElement) document.exitFullscreen(); else w.requestFullscreen && w.requestFullscreen(); });
  const savedCc = storage("studio.captions");
  if (savedCc === false) { state.captions = false; $("#btnCaptions").classList.remove("on"); }
  $("#btnCaptions").addEventListener("click", (e) => { state.captions = !state.captions; storage("studio.captions", state.captions); e.currentTarget.classList.toggle("on", state.captions); renderCaptions(); });
  $("#btnMinimap").addEventListener("click", (e) => { const off = document.getElementById("app").classList.toggle("no-minimap"); e.currentTarget.classList.toggle("on", !off); });
  $$("#abToggle button").forEach((b) => b.addEventListener("click", () => { state.ab = b.dataset.ab; $$("#abToggle button").forEach((x) => x.classList.toggle("on", x === b)); emit("ab", state); loadChunkVideo(true); }));
  $("#btnInsertKp").addEventListener("click", insertKeyposeAtPlayhead);
  $("#btnAddTraj").addEventListener("click", addMoveAtPlayhead);

  on("job", () => { currentUrl = ""; renderHeader(); loadChunkVideo(true); updateButtons(); schedulePreview(); });
  on("mode", () => { currentUrl = ""; renderHeader(); loadChunkVideo(true); updateButtons(); schedulePreview(); });
  on("draft", () => { renderHeader(); updateButtons(); schedulePreview(); });
  on("playhead", (p) => { if (p.source !== "video") syncVideoToPlayhead(); updateReadout(); drawMinimapNow(); renderCaptions(); });
  on("words", renderCaptions);
  on("pending", () => { schedulePreview(); });
  on("pendingLive", () => { schedulePreviewLive(); });
  on("jobStatus", () => renderHeader());
  onLang(() => { renderHeader(); updateReadout(); loadChunkVideo(false); });
  const ro = new ResizeObserver(() => drawMinimap());
  ro.observe($("#minimapWrap"));
  renderHeader();
  updateReadout();
  drawMinimap();
}

// ------------------------------------------------------------------ video source management
function chunkVideoUrl(chunk) {
  if (!chunk) return "";
  if (state.ab === "source" && chunk.source_video_url) return chunk.source_video_url;
  return chunk.video_url || "";
}

function loadChunkVideo(force = false) {
  const v = video();
  const src = timelineSource();
  const chunk = src.kind === "job" ? chunkAt(state.playhead) : null;
  const url = chunkVideoUrl(chunk);
  $("#videoEmpty").classList.toggle("hidden", !!url);
  const isDraft = src.kind === "draft";
  $("#videoEmpty .video-empty-title").textContent = isDraft ? `${t("draft_stage_title")} · ${state.draft.name || ""}` : t("empty_stage_title");
  $("#videoEmpty .video-empty-sub").textContent = isDraft ? t("draft_stage_sub") : t("empty_stage_sub");
  if (!url) $("#videoWrap").classList.remove("loading");
  renderBadge(chunk);
  renderProgressOverlay();
  if (url !== currentUrl || force) {
    currentUrl = url;
    const wasPlaying = state.playing;
    if (url) {
      v.src = url;
      v.load();
      if (wasPlaying) v.play().catch(() => {});
    } else {
      v.pause();
      v.removeAttribute("src");
      v.load();
    }
  }
  const ab = $("#abToggle");
  ab.classList.toggle("hidden", !(chunk && chunk.source_video_url && chunk.video_url && chunk.source_video_url !== chunk.video_url));
}

function renderBadge(chunk) {
  const host = $("#videoBadge");
  host.innerHTML = "";
  const src = timelineSource();
  if (src.kind === "draft") { host.append(el("span", { text: `${t("draft")} · ${state.draft.name || ""}` })); return; }
  if (!chunk) return;
  host.append(el("span", { text: `${t("chunk")} ${chunk.index + 1}/${src.chunks.length}` }));
  if (chunk.edited && state.ab !== "source") host.append(el("span", { class: "edited", text: t("edited") }));
  if (state.ab === "source" && chunk.source_video_url) host.append(el("span", { text: t("source") }));
}

function renderProgressOverlay() {
  const host = $("#videoProgress");
  const job = state.job;
  const active = state.mode === "job" && job && ["queued", "waiting", "running", "starting", "cancelling"].includes(job.state);
  host.classList.toggle("on", !!active);
  if (active) {
    host.querySelector(".bar").style.width = `${Math.round((job.progress || 0) * 100)}%`;
    host.querySelector(".label").textContent = `${job.phase || job.state} · ${Math.round((job.progress || 0) * 100)}% — ${job.message || ""}`;
  }
}

function syncVideoToPlayhead() {
  const v = video();
  const src = timelineSource();
  if (src.kind !== "job") return;
  const chunk = chunkAt(state.playhead);
  const url = chunkVideoUrl(chunk);
  if (url !== currentUrl) { loadChunkVideo(); }
  if (!chunk || !url) return;
  const local = state.playhead - chunk.frame_start;
  const target = (local + 0.5) / src.fps;
  if (Number.isFinite(v.duration) && Math.abs(v.currentTime - target) > 0.5 / src.fps) {
    seekingFromUi = true;
    v.currentTime = Math.min(target, Math.max(0, (v.duration || target) - 0.001));
  }
}

function tick() {
  cancelAnimationFrame(raf);
  const loopFn = () => {
    const v = video();
    if (v.paused) return;
    const src = timelineSource();
    const chunk = chunkAt(state.playhead);
    if (chunk) {
      const frame = chunk.frame_start + Math.floor(v.currentTime * src.fps);
      if (frame !== state.playhead) seek(frame, "video");
    }
    raf = requestAnimationFrame(loopFn);
  };
  raf = requestAnimationFrame(loopFn);
}

function onEnded() {
  const src = timelineSource();
  const chunk = chunkAt(state.playhead);
  const idx = src.chunks.indexOf(chunk);
  if (idx >= 0 && idx < src.chunks.length - 1) {
    seek(src.chunks[idx + 1].frame_start, "ui");
    video().play().catch(() => {});
  } else if (loop && src.chunks.length) {
    seek(0, "ui");
    video().play().catch(() => {});
  } else {
    state.playing = false; emit("playing", state); updatePlayButton();
  }
}

export function togglePlay() {
  const v = video();
  if (!currentUrl) return;
  if (v.paused) v.play().catch(() => {}); else v.pause();
}

export function step(n) {
  const v = video();
  v.pause();
  seek(state.playhead + n, "ui");
}

function updatePlayButton() {
  const btn = $("#btnPlay");
  btn.innerHTML = "";
  btn.append(icon(state.playing ? "pause" : "play"));
}

function updateReadout() {
  const src = timelineSource();
  $("#tcTime").textContent = fmtTime(state.playhead, src.fps);
  $("#tcFrame").textContent = `f ${state.playhead}`;
  const chunk = chunkAt(state.playhead);
  $("#tcChunk").textContent = chunk && src.kind === "job" ? `${t("chunk")} ${chunk.index + 1} · f${state.playhead - chunk.frame_start}` : (src.kind === "draft" ? t("draft") : "");
}

function updateButtons() {
  const src = timelineSource();
  $("#btnInsertKp").disabled = !src.keyposesEnabled;
  $("#btnAddTraj").disabled = !src.totalFrames;
}

export function insertKeyposeAtPlayhead() {
  const src = timelineSource();
  if (!src.keyposesEnabled) { toast(t("draft_hint"), "warn"); return; }
  const id = state.librarySel.keyposes;
  if (!id) { toast(t("select_keypose_first"), "warn"); return; }
  addKeypose(id, state.playhead);
  toast(t("added_keypose"), "ok", 1200);
}

export function addMoveAtPlayhead() {
  const src = timelineSource();
  if (!src.totalFrames) return;
  const start = clamp(state.playhead, 0, src.totalFrames - MINLEN());
  addTraj(start, Math.min(src.totalFrames, start + 36));
  toast(t("added_move"), "ok", 1200);
}
const MINLEN = () => 4;

// ------------------------------------------------------------------ captions (karaoke style)
let capSentence = null;
function renderCaptions() {
  const host = $("#captions");
  if (!state.captions || !state.words || state.mode !== "job") { host.innerHTML = ""; capSentence = null; return; }
  const frame = state.playhead;
  const sent = sentenceAt(frame) || null;
  if (!sent) { host.innerHTML = ""; capSentence = null; return; }
  // long sentences are shown in caption-sized groups of at most 10 words
  const all = state.words.words.filter((w) => w.sentence === sent.i);
  const groupSize = 10;
  const groups = [];
  for (let i = 0; i < all.length; i += groupSize) groups.push(all.slice(i, i + groupSize));
  const group = groups.find((g) => frame < g[g.length - 1].ef) || groups[groups.length - 1];
  const key = `${sent.i}:${group[0].sf}`;
  if (key !== capSentence) {
    capSentence = key;
    host.innerHTML = "";
    const line = el("div", { class: "line" });
    group.forEach((w, i) => { if (i) line.append(" "); line.append(el("span", { class: "w", dataset: { sf: w.sf, ef: w.ef }, text: w.text })); });
    host.append(line);
  }
  host.querySelectorAll(".w").forEach((span) => {
    const sf = Number(span.dataset.sf), ef = Number(span.dataset.ef);
    span.classList.toggle("cur", sf <= frame && frame < ef);
    span.classList.toggle("done", ef <= frame);
  });
}

// ------------------------------------------------------------------ header / crumbs
function renderHeader() {
  const host = $("#stageCrumbs");
  host.innerHTML = "";
  if (state.mode === "draft" && state.draft) {
    host.append(el("span", { class: "pill pill-draft", text: t("draft") }), el("span", { class: "cur", text: state.draft.name || "" }));
    return;
  }
  const job = state.job;
  if (!job) { host.append(el("span", { class: "muted", text: t("no_job_selected") })); return; }
  const chain = [...(job.lineage || [])].reverse();
  chain.forEach((id) => {
    host.append(el("a", { href: "#", text: shortId(id), title: id, onclick: (e) => { e.preventDefault(); emit("openJob", id); } }), el("span", { class: "sep", text: "›" }));
  });
  host.append(el("span", { class: "badge " + (job.action || "generate"), text: job.action || "generate" }), el("span", { class: "cur", text: job.title || shortId(job.job_id), title: job.job_id }));
}

export function shortId(id) {
  return String(id || "").replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-(\w+)$/, "$2-$3 $4:$5 · $7");
}

// ------------------------------------------------------------------ minimap (top-down path)
const schedulePreview = debounce(() => fetchPreview(), 200);
const schedulePreviewLive = debounce(() => fetchPreview(), 350);

function previewBase() {
  const src = timelineSource();
  if (src.kind === "job" && state.job) {
    const chunk = chunkAt(state.playhead);
    if (!chunk || !chunk.npy_path) return null;
    return { base: { kind: "job", job_id: state.job.job_id, chunk_index: chunk.index, frames: chunk.frame_end - chunk.frame_start }, chunk };
  }
  if (src.kind === "draft" && state.draft) {
    const frames = state.draft.frames || 0;
    if (!frames) return null;
    const id = state.draft.trajectoryId || $("#genTrajectory").value || "";
    return { base: id ? { kind: "trajectory", id, frames } : { kind: "still", frames }, chunk: { frame_start: 0, frame_end: frames, index: 0 } };
  }
  return null;
}

async function fetchPreview() {
  const spec = previewBase();
  if (!spec) { state.minimapPreview = null; drawMinimap(); return; }
  const { base, chunk } = spec;
  // trajectory segments are global frames; the base trajectory restarts per chunk
  const script = state.pending.traj
    .map((s) => ({ ...s, start: s.start - chunk.frame_start, end: s.end - chunk.frame_start }))
    .filter((s) => s.end > 0 && s.start < chunk.frame_end - chunk.frame_start)
    .map((s) => ({ ...s, start: Math.max(0, s.start), end: Math.min(chunk.frame_end - chunk.frame_start, s.end) }));
  const token = ++previewToken;
  try {
    const res = await api.trajPreview(base, script);
    if (token !== previewToken) return;
    state.minimapPreview = { ...res, chunkStart: chunk.frame_start };
    drawMinimap();
  } catch (err) {
    if (token === previewToken) { state.minimapPreview = null; drawMinimap(); }
  }
}

function drawMinimap() {
  const cv = $("#minimap");
  const wrap = $("#minimapWrap");
  const size = Math.max(120, Math.min(wrap.clientWidth || 260, 320));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = size * dpr; cv.height = size * dpr;
  cv.style.width = `${size}px`; cv.style.height = `${size}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const pv = state.minimapPreview;
  // grid
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 6; i += 1) { const p = (size / 6) * i; ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke(); }
  if (!pv) {
    ctx.fillStyle = "#3a4353"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(t("no_job_selected"), size / 2, size / 2);
    return;
  }
  const xs = [...pv.base.x, ...pv.result.x], zs = [...pv.base.z, ...pv.result.z];
  let minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const span = Math.max(maxX - minX, maxZ - minZ, 1.0);
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const pad = 22;
  const scale = (size - pad * 2) / span;
  // screen: x right = +x (character left side is +x, so mirror so that left is left when facing away from us)
  const sx = (x) => size / 2 - (x - cx) * scale;
  const sz = (z) => size / 2 - (z - cz) * scale;
  // scale bar (1 m)
  ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.beginPath(); ctx.moveTo(10, size - 10); ctx.lineTo(10 + scale, size - 10); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText("1 m", 12, size - 14);
  ctx.textAlign = "right"; ctx.fillText("▲ forward (+z)", size - 8, 14);
  const drawPath = (p, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.beginPath();
    p.x.forEach((x, i) => { const X = sx(x), Z = sz(p.z[i]); if (i === 0) ctx.moveTo(X, Z); else ctx.lineTo(X, Z); });
    ctx.stroke();
  };
  drawPath(pv.base, "rgba(127,138,156,0.7)", 1.5);
  if (state.pending.traj.length) drawPath(pv.result, "#f5a524", 2.2);
  // start marker
  ctx.fillStyle = "#2ecc71"; ctx.beginPath(); ctx.arc(sx(pv.result.x[0]), sz(pv.result.z[0]), 3, 0, Math.PI * 2); ctx.fill();
  drawMinimapNow(ctx, sx, sz, pv);
}

function drawMinimapNow(ctx, sx, sz, pv) {
  if (!ctx) { drawMinimap(); return; }
  const p = state.pending.traj.length ? pv.result : pv.base;
  const local = state.playhead - (pv.chunkStart || 0);
  if (local < 0 || local >= p.frames) return;
  // nearest downsampled index
  let lo = 0, hi = p.index.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (p.index[mid] < local) lo = mid + 1; else hi = mid; }
  const i = lo;
  const X = sx(p.x[i]), Z = sz(p.z[i]);
  const yaw = p.yaw[i];
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X, Z);
  ctx.lineTo(X - Math.sin(yaw) * 12, Z - Math.cos(yaw) * 12); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(X, Z, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.6)"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(`y ${p.y[i] >= 0 ? "+" : ""}${p.y[i].toFixed(2)} m`, X + 8, Z + 4);
}
