// Central application state with a minimal pub/sub.
import { storage, uid } from "./util.js";

const listeners = new Map();

export const state = {
  config: window.STUDIO_CONFIG || {},
  system: null,
  assets: { keyposes: [], trajectories: [], identities: [] },
  assetMeta: { keyposes: { categories: [], speakers: [] }, trajectories: { categories: [], speakers: [], frequencies: [], default: "" } },
  jobs: [],
  jobsVersion: -1,
  job: null,                 // detail of the selected job
  draft: null,               // { file, name, frames, peaks, trajectoryId } while preparing a generation
  mode: "job",               // "job" | "draft" -> what the timeline shows
  playhead: 0,               // global frame
  playing: false,
  ab: "edited",
  selection: null,           // { type: "kp" | "tr", id }
  pending: { keyposes: [], traj: [] },
  libraryTab: "keyposes",
  identityId: "",             // identity selected in the generate form; trajectories follow it
  trajAllSpeakers: false,     // show the trajectories of all speakers
  librarySel: { keyposes: null, trajectories: null, identities: null },
  inspectorTab: "edit",
  zoom: 0,                   // 0..100 slider value
  minimapPreview: null,      // last /api/traj/preview result
  editOptions: { regen_windows: 3, render: true },
  words: null,               // { words: [...], sentences: [...] } for the current job (root job's transcript)
  captions: true,
};

// binary search helpers over word / sentence spans ([sf, ef) frames, sorted by sf)
function spanAt(list, frame) {
  if (!list || !list.length) return null;
  let lo = 0, hi = list.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (list[mid].sf <= frame) lo = mid; else hi = mid - 1; }
  const item = list[lo];
  return item && item.sf <= frame && frame < item.ef ? item : null;
}
export function wordAt(frame) { return state.words ? spanAt(state.words.words, frame) : null; }
export function sentenceAt(frame) { return state.words ? spanAt(state.words.sentences, frame) : null; }
export function nearestWord(frame, maxGap = 15) {
  if (!state.words || !state.words.words.length) return null;
  const exact = wordAt(frame);
  if (exact) return exact;
  let best = null, bestD = Infinity;
  for (const w of state.words.words) { const d = frame < w.sf ? w.sf - frame : frame - w.ef; if (d < bestD) { bestD = d; best = w; } }
  return bestD <= maxGap ? best : null;
}

// Move the playhead.  ``source`` lets the stage ignore its own video-driven updates.
export function seek(frame, source = "ui") {
  const src = timelineSource();
  const max = Math.max(0, src.totalFrames - 1);
  state.playhead = Math.max(0, Math.min(max, Math.round(frame)));
  emit("playhead", { frame: state.playhead, source });
}

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, payload) {
  (listeners.get(event) || []).forEach((fn) => { try { fn(payload); } catch (err) { console.error(`[store:${event}]`, err); } });
}

export function set(patch, event) {
  Object.assign(state, patch);
  if (event) emit(event, state);
}

// ---- helpers around the timeline source
export function timelineSource() {
  if (state.mode === "draft" && state.draft) {
    return {
      kind: "draft",
      totalFrames: state.draft.frames || 0,
      fps: state.config.fps || 30,
      chunks: state.draft.frames ? [{ index: 0, frame_start: 0, frame_end: state.draft.frames, frames: state.draft.frames, draft: true }] : [],
      audioPeaks: state.draft.peaks || null,
      keyposesEnabled: false,
    };
  }
  const job = state.job;
  if (!job) return { kind: "none", totalFrames: 0, fps: state.config.fps || 30, chunks: [], keyposesEnabled: false };
  return {
    kind: "job",
    totalFrames: job.total_frames || 0,
    fps: job.fps || state.config.fps || 30,
    chunks: job.chunks || [],
    keyposesEnabled: job.state === "succeeded" && (job.chunks || []).some((c) => c.npy_path),
  };
}

export function chunkAt(frame) {
  const src = timelineSource();
  return src.chunks.find((c) => frame >= c.frame_start && frame < c.frame_end) || src.chunks[src.chunks.length - 1] || null;
}

// ---- pending edits (persisted per job / draft)
function pendingKey() {
  if (state.mode === "draft") return "studio.pending.draft";
  return state.job ? `studio.pending.${state.job.job_id}` : null;
}

export function loadPending() {
  const key = pendingKey();
  const saved = key ? storage(key) : null;
  state.pending = saved && typeof saved === "object" ? { keyposes: saved.keyposes || [], traj: saved.traj || [] } : { keyposes: [], traj: [] };
  emit("pending", state);
}

export function savePending() {
  const key = pendingKey();
  if (key) storage(key, state.pending);
  emit("pending", state);
}

export function clearPending() {
  const key = pendingKey();
  if (key) storage(key, null);
  state.pending = { keyposes: [], traj: [] };
  state.selection = null;
  emit("pending", state);
  emit("selection", state);
}

export function addKeypose(keyposeId, frame, extra = {}) {
  const item = { id: uid("kp"), keypose_id: keyposeId, frame: Math.round(frame), part: "full_body", strength: 1, sigma: 6, bands: ["ca3", "cd3", "cd2", "cd1"], ...extra };
  state.pending.keyposes.push(item);
  state.pending.keyposes.sort((a, b) => a.frame - b.frame);
  savePending();
  select("kp", item.id);
  return item;
}

export function addTraj(start, end, extra = {}) {
  const prim = state.config.primitives || {};
  const type = extra.type || "forward";
  const item = { id: uid("tr"), start: Math.round(start), end: Math.round(end), type, amount: prim[type] ? prim[type].default : 1.2, steps: type === "forward" || type === "backward" ? 2 : undefined, mode: "replace", hold: 0.4, ...extra };
  if (item.steps !== undefined) item.amount = item.steps * (state.config.step_length || 0.6);
  state.pending.traj.push(item);
  state.pending.traj.sort((a, b) => a.start - b.start);
  savePending();
  select("tr", item.id);
  return item;
}

export function updatePending(type, id, patch) {
  const list = type === "kp" ? state.pending.keyposes : state.pending.traj;
  const item = list.find((x) => x.id === id);
  if (!item) return null;
  Object.assign(item, patch);
  if (type === "kp") list.sort((a, b) => a.frame - b.frame);
  else list.sort((a, b) => a.start - b.start);
  savePending();
  return item;
}

export function removePending(type, id) {
  if (type === "kp") state.pending.keyposes = state.pending.keyposes.filter((x) => x.id !== id);
  else state.pending.traj = state.pending.traj.filter((x) => x.id !== id);
  if (state.selection && state.selection.id === id) state.selection = null;
  savePending();
  emit("selection", state);
}

export function select(type, id) {
  state.selection = type ? { type, id } : null;
  emit("selection", state);
}

export function selectedItem() {
  const sel = state.selection;
  if (!sel) return null;
  const list = sel.type === "kp" ? state.pending.keyposes : state.pending.traj;
  return list.find((x) => x.id === sel.id) || null;
}

export function pendingCount() {
  return state.pending.keyposes.length + state.pending.traj.length;
}

export function asset(kind, id) {
  return (state.assets[kind] || []).find((x) => x.id === id) || null;
}

// How much a person moves is mostly a property of the speaker, so a trajectory belongs to an identity;
// another speaker's walking style fits neither the body size nor what the model saw for this identity.
export function identitySpeakerKey() {
  const ident = state.identityId ? asset("identities", state.identityId) : null;
  return ident ? (ident.speaker_key || "") : "";
}

export function ownTrajectories(speakerKey = null) {
  const key = speakerKey === null ? identitySpeakerKey() : speakerKey;
  const all = state.assets.trajectories || [];
  return key ? all.filter((t) => t.speaker_key === key) : all;
}

/** The calmest trajectory of the identity's speaker; the library default without an identity. */
export function quietestOwnTrajectory() {
  const own = ownTrajectories();
  if (!own.length) return "";
  if (!identitySpeakerKey()) return state.config.default_trajectory || own[0].id;
  return [...own].sort((a, b) => (a.amplitude_score ?? 0) - (b.amplitude_score ?? 0))[0].id;
}
