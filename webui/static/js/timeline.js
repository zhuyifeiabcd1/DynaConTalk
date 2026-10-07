// Timeline: ruler, chunk lane, waveform, keypose markers, trajectory segments, playhead.
import { $, el, icon, clamp, fmtTime, toast, throttle } from "./util.js";
import { t, onLang } from "./i18n.js";
import { state, on, emit, seek, timelineSource, chunkAt, addKeypose, addTraj, updatePending, removePending, select, savePending, asset, pendingCount, wordAt, sentenceAt } from "./store.js";
import { api } from "./api.js";

const MIN_SEG = 4;
let ppf = 1;              // pixels per frame
let viewportW = 800;
let peaks = null;         // Float32Array of per-frame peaks for the audio lane
let peaksKey = null;
const peaksCache = new Map();
let raf = 0;

const scroll = () => $("#tlScroll");
const canvas = () => $("#tlCanvas");

export function initTimeline() {
  const sc = scroll();
  const ro = new ResizeObserver(() => { viewportW = sc.clientWidth; layout(); });
  ro.observe(sc);
  viewportW = sc.clientWidth;

  $("#tlZoom").addEventListener("input", (e) => { state.zoom = Number(e.target.value); layout(); });
  $("#tlZoomIn").addEventListener("click", () => setZoom(state.zoom + 12));
  $("#tlZoomOut").addEventListener("click", () => setZoom(state.zoom - 12));
  $("#tlFit").addEventListener("click", () => setZoom(0));
  sc.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const before = frameAtClient(e.clientX);
      setZoom(state.zoom - Math.sign(e.deltaY) * 8, false);
      // keep the frame under the cursor stationary
      const rect = sc.getBoundingClientRect();
      sc.scrollLeft = before * ppf - (e.clientX - rect.left);
    } else if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      sc.scrollLeft += e.deltaY;
      e.preventDefault();
    }
  }, { passive: false });

  // seek by pressing on empty timeline areas
  canvas().addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (e.target.closest(".kp-marker, .tr-seg")) return;
    const move = (ev) => seek(frameAtClient(ev.clientX), "timeline");
    move(e);
    if (!e.target.closest(".tl-track.tr, .tl-track.kp")) select(null);
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });

  // keypose drag & drop from the library
  const kpTrack = $("#trackKp");
  kpTrack.addEventListener("dragover", (e) => {
    if (!timelineSource().keyposesEnabled) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    kpTrack.classList.add("drop");
    seek(frameAtClient(e.clientX), "timeline");
  });
  kpTrack.addEventListener("dragleave", () => kpTrack.classList.remove("drop"));
  kpTrack.addEventListener("drop", (e) => {
    e.preventDefault();
    kpTrack.classList.remove("drop");
    const id = e.dataTransfer.getData("text/keypose");
    if (!id || !timelineSource().keyposesEnabled) return;
    addKeypose(id, frameAtClient(e.clientX));
    toast(t("added_keypose"), "ok", 1200);
  });
  // double-click on the trajectory lane adds a move there
  $("#trackTr").addEventListener("dblclick", (e) => {
    const src = timelineSource();
    if (!src.totalFrames) return;
    const f = frameAtClient(e.clientX);
    addTraj(f, Math.min(src.totalFrames, f + 36));
  });

  on("job", () => { state.zoom = 0; $("#tlZoom").value = 0; ensurePeaks(); layout(); });
  on("draft", () => { ensurePeaks(); layout(); });
  on("mode", () => { ensurePeaks(); layout(); });
  on("pending", () => renderTracks());
  on("selection", () => renderTracks());
  on("assets", () => renderTracks());
  on("playhead", (p) => { movePlayhead(p.frame, p.source !== "video"); renderChunkHighlight(); highlightWord(p.frame); });
  on("words", () => renderWords());
  on("jobStatus", () => renderWordsEmpty());
  onLang(() => layout());
  $("#btnTranscribe").addEventListener("click", requestTranscribe);
  layout();
}

export async function requestTranscribe() {
  if (!state.job) return;
  try {
    const status = await api.transcribe(state.job.job_id);
    toast(`${t("words_queued")} · ${status.job_id}`, "ok");
    $("#btnTranscribe").disabled = true;
    renderWordsEmpty(true);
  } catch (err) { toast(err.message, "err"); }
}

export function setZoom(value, updateSlider = true) {
  state.zoom = clamp(value, 0, 100);
  if (updateSlider) $("#tlZoom").value = state.zoom;
  layout();
}

function fitPpf() {
  const total = Math.max(1, timelineSource().totalFrames);
  return Math.max(0.02, (viewportW - 2) / total);
}

function computePpf() {
  const min = fitPpf();
  const max = Math.max(min, 10);
  return min * Math.pow(max / min, state.zoom / 100);
}

export function frameAtClient(clientX) {
  const sc = scroll();
  const rect = sc.getBoundingClientRect();
  const x = clientX - rect.left + sc.scrollLeft;
  const total = timelineSource().totalFrames;
  return clamp(Math.round(x / ppf), 0, Math.max(0, total - 1));
}

function layout() {
  ppf = computePpf();
  const src = timelineSource();
  const width = Math.max(viewportW, Math.round(src.totalFrames * ppf));
  canvas().style.width = `${width}px`;
  renderRuler(width);
  renderChunks();
  renderWaveform(width);
  renderTracks();
  renderWords();
  movePlayhead(state.playhead, true);
}

// ------------------------------------------------------------------ words lane
let wordNodes = [];        // [{node, sf, ef}] in the lane, sorted by sf
let wordMode = "words";
let curWordNode = null;

function renderWordsEmpty(pending = false) {
  const track = $("#trackWords");
  const btn = $("#btnTranscribe");
  const src = timelineSource();
  const job = state.job;
  const isJob = src.kind === "job" && job;
  const busy = pending || (isJob && job.transcribe_pending);
  btn.classList.toggle("hidden", !(isJob && !state.words));
  btn.disabled = !!busy;
  if (!state.words) {
    track.innerHTML = "";
    if (isJob) track.append(el("div", { class: "words-empty" }, [busy ? t("transcribing") : t("no_words_hint")]));
  }
}

function renderWords() {
  const track = $("#trackWords");
  track.innerHTML = "";
  wordNodes = [];
  curWordNode = null;
  if (!state.words || state.mode !== "job") { renderWordsEmpty(); return; }
  renderWordsEmpty();
  const words = state.words.words || [];
  const sentences = state.words.sentences || [];
  if (!words.length) { track.append(el("div", { class: "words-empty", text: "—" })); return; }
  const avgPx = words.reduce((a, w) => a + (w.ef - w.sf), 0) / words.length * ppf;
  wordMode = avgPx >= 18 ? "words" : "sentences";
  const list = wordMode === "words" ? words : sentences;
  const frag = document.createDocumentFragment();
  list.forEach((w) => {
    const width = Math.max(4, (w.ef - w.sf) * ppf - 2);
    const fits = width >= w.text.length * 6.4 + 10 || (wordMode === "sentences" && width > 40);
    const node = el("div", { class: `${wordMode === "words" ? "word" : "sentence"} ${fits ? "" : "bare"}`, style: { left: `${w.sf * ppf + 1}px`, width: `${width}px` }, title: `${w.text}\nf${w.sf}–${w.ef - 1} · ${fmtTime(w.sf, timelineSource().fps)}`, text: fits ? w.text : "" });
    node.addEventListener("click", (e) => { e.stopPropagation(); seek(w.sf, "timeline"); });
    frag.append(node);
    wordNodes.push({ node, sf: w.sf, ef: w.ef });
  });
  track.append(frag);
  highlightWord(state.playhead);
}

function highlightWord(frame) {
  if (!wordNodes.length) return;
  let lo = 0, hi = wordNodes.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (wordNodes[mid].sf <= frame) lo = mid; else hi = mid - 1; }
  const hit = wordNodes[lo] && wordNodes[lo].sf <= frame && frame < wordNodes[lo].ef ? wordNodes[lo].node : null;
  if (hit === curWordNode) return;
  if (curWordNode) curWordNode.classList.remove("cur");
  curWordNode = hit;
  if (hit) hit.classList.add("cur");
}

// ------------------------------------------------------------------ ruler & chunks
function renderRuler(width) {
  const host = $("#tlRuler");
  host.innerHTML = "";
  const src = timelineSource();
  const fps = src.fps;
  if (!src.totalFrames) return;
  // choose a label interval so labels are >= 70px apart
  const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
  let stepSec = candidates.find((s) => s * fps * ppf >= 70) || 60;
  const minor = stepSec / 5;
  for (let s = 0; s <= src.totalFrames / fps + 1e-6; s += minor) {
    const isMajor = Math.abs(s / stepSec - Math.round(s / stepSec)) < 1e-6;
    const x = s * fps * ppf;
    if (x > width) break;
    const tick = el("div", { class: `tick ${isMajor ? "" : "minor"}`, style: { left: `${x}px` } });
    if (isMajor) tick.textContent = fmtLabel(s);
    host.append(tick);
  }
}

function fmtLabel(sec) {
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return m ? `${m}:${s.toFixed(0).padStart(2, "0")}` : `${s % 1 ? s.toFixed(1) : s.toFixed(0)}s`;
}

function renderChunks() {
  const host = $("#trackChunks");
  host.innerHTML = "";
  const src = timelineSource();
  src.chunks.forEach((c, i) => {
    const block = el("div", {
      class: `chunk-block ${c.edited ? "edited" : ""}`,
      style: { left: `${c.frame_start * ppf}px`, width: `${Math.max(2, (c.frame_end - c.frame_start) * ppf - 2)}px` },
      title: `${t("chunk")} ${c.index + 1}: f${c.frame_start}–${c.frame_end - 1}`,
      onclick: (e) => { e.stopPropagation(); seek(c.frame_start, "timeline"); },
    }, [
      c.draft ? t("draft") : `${t("chunk")} ${c.index + 1}/${src.chunks.length}`,
      el("span", { class: "mono", text: `${fmtTime(c.frame_end - c.frame_start, src.fps)}` }),
    ]);
    host.append(block);
  });
  renderChunkHighlight();
}

function renderChunkHighlight() {
  const cur = chunkAt(state.playhead);
  $("#trackChunks").querySelectorAll(".chunk-block").forEach((b, i) => {
    const src = timelineSource();
    b.classList.toggle("cur", !!cur && src.chunks[i] === cur);
  });
}

// ------------------------------------------------------------------ audio lane
export async function decodePeaks(arrayBuffer, fps) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  const ctx = new Ctx();
  try {
    const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const data = buf.getChannelData(0);
    const per = buf.sampleRate / fps;
    const frames = Math.floor(buf.duration * fps);
    const out = new Float32Array(frames);
    for (let f = 0; f < frames; f += 1) {
      const a = Math.floor(f * per), b = Math.min(data.length, Math.floor((f + 1) * per));
      let m = 0;
      for (let i = a; i < b; i += 4) { const v = Math.abs(data[i]); if (v > m) m = v; }
      out[f] = m;
    }
    return { peaks: out, frames, duration: buf.duration };
  } finally { ctx.close(); }
}

async function ensurePeaks() {
  const src = timelineSource();
  let key = null;
  if (src.kind === "draft") { peaks = state.draft.peaks || null; peaksKey = "draft"; renderWaveform(); return; }
  if (src.kind === "job" && state.job && state.job.audio_url) key = state.job.audio_url;
  if (!key) { peaks = null; peaksKey = null; renderWaveform(); return; }
  if (peaksKey === key && peaks) return;
  peaksKey = key;
  if (peaksCache.has(key)) { peaks = peaksCache.get(key); renderWaveform(); return; }
  peaks = null;
  renderWaveform();
  try {
    const res = await fetch(key);
    const buf = await res.arrayBuffer();
    const decoded = await decodePeaks(buf, src.fps);
    if (decoded) { peaksCache.set(key, decoded.peaks); if (peaksKey === key) { peaks = decoded.peaks; renderWaveform(); } }
  } catch (err) { console.warn("waveform", err); }
}

function renderWaveform(width) {
  const cv = $("#waveform");
  const track = $("#trackAudio");
  width = width || canvas().clientWidth;
  const w = Math.min(16000, Math.max(1, Math.round(width)));
  const h = track.clientHeight || 40;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!peaks) {
    const src = timelineSource();
    if (src.totalFrames) { ctx.fillStyle = "#3a4353"; ctx.font = "10px sans-serif"; ctx.fillText(state.job && state.job.audio_url ? "…" : "", 6, 14); }
    return;
  }
  const scaleX = w / width;   // canvas may be capped
  const mid = h / 2;
  ctx.fillStyle = "rgba(127, 138, 156, 0.55)";
  const framesPerPx = 1 / (ppf * scaleX);
  for (let x = 0; x < w; x += 1) {
    const f0 = Math.floor(x * framesPerPx), f1 = Math.max(f0 + 1, Math.floor((x + 1) * framesPerPx));
    let m = 0;
    for (let f = f0; f < f1 && f < peaks.length; f += 1) if (peaks[f] > m) m = peaks[f];
    const hh = Math.max(1, m * (h - 6));
    ctx.fillRect(x, mid - hh / 2, 1, hh);
  }
}

// ------------------------------------------------------------------ keyposes & trajectory lanes
function renderTracks() {
  const src = timelineSource();
  const kpTrack = $("#trackKp"), trTrack = $("#trackTr");
  kpTrack.innerHTML = ""; trTrack.innerHTML = "";
  kpTrack.classList.toggle("disabled", !src.keyposesEnabled);
  trTrack.classList.toggle("disabled", !src.totalFrames);
  const sel = state.selection;
  // committed (already part of the job request) – read only
  if (src.kind === "job" && state.job) {
    // edits inherited from ancestor jobs first: they are already in the motion but not in this job's
    // request; without them the timeline would only show the latest edit
    (state.job.inherited_keyposes || []).forEach((k) => kpTrack.append(keyposeMarker({ ...k, id: `i_${k.from_job}_${k.id || k.frame}` }, true, false, true)));
    (state.job.inherited_traj || []).forEach((s) => trTrack.append(trajSegment({ ...s, id: `i_${s.from_job}_${s.id || s.start}` }, true, false, true)));
    (state.job.keyposes || []).forEach((k) => kpTrack.append(keyposeMarker({ ...k, id: `c_${k.id || k.frame}` }, true, false)));
    (state.job.traj_script || []).forEach((s) => trTrack.append(trajSegment({ ...s, id: `c_${s.id || s.start}` }, true, false)));
  }
  state.pending.keyposes.forEach((k) => kpTrack.append(keyposeMarker(k, false, sel && sel.type === "kp" && sel.id === k.id)));
  state.pending.traj.forEach((s) => trTrack.append(trajSegment(s, false, sel && sel.type === "tr" && sel.id === s.id)));
  renderPendingBadge();
}

function renderPendingBadge() {
  const host = $("#tlPending");
  host.innerHTML = "";
  const nk = state.pending.keyposes.length, nt = state.pending.traj.length;
  if (nk) host.append(el("span", { class: "n kp" }, [icon("pose"), String(nk)]));
  if (nt) host.append(el("span", { class: "n tr" }, [icon("walk"), String(nt)]));
  const n = pendingCount();
  const draft = state.mode === "draft";
  $("#btnApply").classList.toggle("hidden", draft);
  $("#btnApply").disabled = !n || draft || !state.job || state.job.state !== "succeeded";
  $("#btnDiscard").classList.toggle("hidden", !n);
  if (draft && nt) host.append(el("span", { class: "muted", text: t("generated_with_job") }));
}

function keyposeMarker(k, committed, on, inherited = false) {
  const kp = asset("keyposes", k.keypose_id);
  const from = inherited ? `\n${t("from_job")}: ${String(k.from_job || "").slice(9, 21)}` : "";
  const node = el("div", { class: `kp-marker ${committed ? "committed" : ""} ${inherited ? "inherited" : ""} ${on ? "on" : ""}`, style: { left: `${k.frame * ppf}px` }, title: `${kp ? kp.name : k.keypose_id} · f${k.frame} · ${k.part}${from}${k.reason ? `\n${k.reason}` : ""}` });
  node.append(
    el("div", { class: "band", style: { width: `${Math.max(2, 4 * (k.sigma || 6) * ppf)}px` } }),
    el("div", { class: "line" }),
    el("div", { class: "pin" }, [kp && kp.media_url ? el("img", { src: kp.media_url, alt: "" }) : null, el("div", { class: "part", text: shortPart(k.part) })]),
  );
  if (committed) return node;
  node.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    select("kp", k.id);
    const startX = e.clientX, startFrame = k.frame;
    let moved = false;
    const total = timelineSource().totalFrames;
    const move = throttle((ev) => {
      const df = (ev.clientX - startX) / ppf;
      const f = clamp(Math.round(startFrame + df), 0, total - 1);
      if (f !== k.frame) { moved = true; k.frame = f; node.style.left = `${f * ppf}px`; node.title = `f${f}`; seek(f, "timeline"); emit("pendingLive", state); }
    }, 16);
    const up = () => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      if (moved) { updatePending("kp", k.id, { frame: k.frame }); } else { seek(k.frame, "timeline"); }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  return node;
}

function shortPart(part) {
  return ({ right_arm: "R", left_arm: "L", both_arms: "LR", hands: "H", torso: "T", upper_body: "UB", full_body: "FB" })[part] || "";
}

export function primIcon(type) {
  if (type.startsWith("turn")) return "turn";
  if (type === "crouch" || type === "rise") return "crouch";
  if (type === "hold") return "pause";
  return "walk";
}

export function segLabel(s) {
  const prim = (state.config.primitives || {})[s.type] || {};
  const name = t(s.type === "hold" ? "hold_still" : s.type);
  if (prim.channel === null || prim.channel === undefined) return name;
  if (prim.unit === "deg") return `${name} ${Math.round(s.amount)}°`;
  if (s.steps !== undefined && s.steps !== null && s.steps !== "") return `${name} ${Number(s.steps)} ${t("steps")}`;
  return `${name} ${Number(s.amount).toFixed(2)} m`;
}

function trajSegment(s, committed, on, inherited = false) {
  const node = el("div", {
    class: `tr-seg ${committed ? "committed" : ""} ${inherited ? "inherited" : ""} ${on ? "on" : ""}`,
    style: { left: `${s.start * ppf}px`, width: `${Math.max(6, (s.end - s.start) * ppf)}px` },
    title: `${segLabel(s)} · f${s.start}–${s.end - 1} · ${fmtTime(s.end - s.start, timelineSource().fps)}`,
  }, [icon(primIcon(s.type)), el("span", { text: segLabel(s) })]);
  if (committed) return node;
  node.append(el("div", { class: "handle l" }), el("div", { class: "handle r" }));
  node.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    select("tr", s.id);
    const total = timelineSource().totalFrames;
    const handle = e.target.closest(".handle");
    const mode = handle ? (handle.classList.contains("l") ? "l" : "r") : "move";
    const startX = e.clientX, s0 = s.start, e0 = s.end;
    let moved = false;
    const move = throttle((ev) => {
      const df = Math.round((ev.clientX - startX) / ppf);
      let ns = s0, ne = e0;
      if (mode === "move") { ns = clamp(s0 + df, 0, total - (e0 - s0)); ne = ns + (e0 - s0); }
      else if (mode === "l") { ns = clamp(s0 + df, 0, e0 - MIN_SEG); }
      else { ne = clamp(e0 + df, s0 + MIN_SEG, total); }
      if (ns !== s.start || ne !== s.end) {
        moved = true; s.start = ns; s.end = ne;
        node.style.left = `${ns * ppf}px`; node.style.width = `${Math.max(6, (ne - ns) * ppf)}px`;
        seek(mode === "r" ? ne - 1 : ns, "timeline");
        emit("pendingLive", state);
      }
    }, 16);
    const up = () => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      if (moved) updatePending("tr", s.id, { start: s.start, end: s.end }); else seek(s.start, "timeline");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  return node;
}

// ------------------------------------------------------------------ playhead
function movePlayhead(frame, ensureVisible) {
  const x = frame * ppf;
  $("#playhead").style.left = `${x}px`;
  if (ensureVisible) {
    const sc = scroll();
    if (x < sc.scrollLeft + 20 || x > sc.scrollLeft + viewportW - 20) sc.scrollLeft = Math.max(0, x - viewportW * 0.3);
  } else {
    // during playback keep the playhead in view with a soft page flip
    const sc = scroll();
    if (x > sc.scrollLeft + viewportW - 10) sc.scrollLeft = x - 40;
    else if (x < sc.scrollLeft) sc.scrollLeft = Math.max(0, x - 40);
  }
}

export function removeSelected() {
  const sel = state.selection;
  if (!sel) return false;
  removePending(sel.type, sel.id);
  return true;
}
