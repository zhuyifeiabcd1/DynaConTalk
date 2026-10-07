// Inspector: Generate form, Edit properties (selected marker / segment, pending list, apply), Info.
import { $, $$, el, icon, toast, fmtTime, fmtDuration, fmtDate, fmtBytes, clamp, confirm, modal, storage } from "./util.js";
import { t, onLang } from "./i18n.js";
import { api } from "./api.js";
import { state, on, emit, set, timelineSource, selectedItem, updatePending, removePending, select, clearPending, pendingCount, asset, savePending, seek, loadPending, nearestWord, sentenceAt, addKeypose, identitySpeakerKey, ownTrajectories, quietestOwnTrajectory } from "./store.js";
import { requestTranscribe } from "./timeline.js";
import { decodePeaks, segLabel, primIcon } from "./timeline.js";
import { displayName, catLabel } from "./library.js";
import { selectJob, refreshJobs } from "./jobs.js";
import { shortId } from "./stage.js";

export function initInspector() {
  $$("#inspTabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  initGenerateForm();
  on("selection", renderEdit);
  on("pending", renderEdit);
  on("job", () => { renderEdit(); renderInfo(); syncReuseAudio(); });
  on("jobStatus", () => { renderInfo(); });
  on("draft", renderEdit);
  on("mode", renderEdit);
  on("assets", () => { fillIdentity(); applyDefaultTrajectory(); renderTrajPicker(); renderEdit(); });
  on("words", () => { renderEdit(); renderInfo(); });
  on("playhead", () => { const cur = $("#inspInfoBody .transcript"); if (cur) highlightTranscript(); });
  on("pickTrajectory", (id) => { setTrajectory(id); });
  // setting select.value does not fire "change", so run the identity switch here; otherwise picking
  // a speaker from the library would not switch the trajectory
  on("pickIdentity", (id) => { $("#genIdentity").value = id; onIdentityChange(true); showTab("generate"); });
  onLang(() => { renderEdit(); renderInfo(); renderTrajPicker(); fillIdentity(); });
  renderEdit();
  renderInfo();
}

export function showTab(name) {
  state.inspectorTab = name;
  $$("#inspTabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $$(".insp-page").forEach((p) => p.classList.toggle("on", p.id === `insp${name[0].toUpperCase()}${name.slice(1)}`));
  const wantDraft = name === "generate" && !!state.draft;
  const mode = wantDraft ? "draft" : "job";
  if (mode !== state.mode) { state.mode = mode; state.selection = null; loadPending(); emit("mode", state); seek(0, "ui"); }
}

// ------------------------------------------------------------------ generate
function initGenerateForm() {
  const form = $("#genForm");
  const input = $("#audioInput");
  const drop = $("#audioDrop");
  input.addEventListener("change", () => { if (input.files && input.files[0]) setAudioFile(input.files[0]); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  $("#reuseAudio").addEventListener("change", (e) => { if (e.target.checked) reuseJobAudio(); });
  $("#genTrajPick").addEventListener("click", () => { emit("libraryTab", "trajectories"); });
  $("#genIdentity").addEventListener("change", () => onIdentityChange(true));
  form.addEventListener("submit", submitGenerate);
  fillIdentity();
  renderTrajPicker();
}

export async function setAudioFile(file) {
  if (!file) return;
  const fps = state.config.fps || 30;
  const drop = $("#audioDrop");
  $("#audioName").textContent = `${file.name} · ${fmtBytes(file.size)}`;
  drop.classList.add("has");
  $("#reuseAudio").checked = false;
  let peaks = null, frames = 0;
  try {
    const decoded = await decodePeaks(await file.arrayBuffer(), fps);
    if (decoded) { peaks = decoded.peaks; frames = decoded.frames; $("#audioName").textContent += ` · ${fmtTime(frames, fps)} · ${frames} f`; }
  } catch (err) { console.warn("decode", err); }
  state.draft = { file, name: file.name, frames, peaks, trajectoryId: $("#genTrajectory").value || "", fromJob: null };
  emit("draft", state);
  showTab("generate");
  emit("mode", state);
}

async function reuseJobAudio() {
  const job = state.job;
  if (!job || !job.audio_url) { $("#reuseAudio").checked = false; toast(t("no_audio"), "warn"); return; }
  const fps = state.config.fps || 30;
  $("#audioName").textContent = `${job.audio_name || job.job_id} (${t("reuse_audio")})`;
  $("#audioDrop").classList.add("has");
  $("#audioInput").value = "";
  let peaks = null, frames = job.total_frames || 0;
  try { const buf = await (await fetch(job.audio_url)).arrayBuffer(); const d = await decodePeaks(buf, fps); if (d) { peaks = d.peaks; frames = d.frames; } } catch (_) {}
  state.draft = { file: null, name: job.audio_name || job.job_id, frames, peaks, trajectoryId: $("#genTrajectory").value || "", fromJob: job.job_id };
  emit("draft", state);
  showTab("generate");
  emit("mode", state);
}

function syncReuseAudio() {
  $("#reuseAudioWrap").classList.toggle("hidden", !(state.job && state.job.audio_url));
}

function fillIdentity() {
  const sel = $("#genIdentity");
  const cur = sel.value;
  sel.innerHTML = "";
  sel.append(el("option", { value: "", text: t("auto") }));
  (state.assets.identities || []).forEach((x) => sel.append(el("option", { value: x.id, text: `${displayName(x)}${x.speaker_id !== undefined ? ` · #${x.speaker_id}` : ""}` })));
  sel.value = cur;
  onIdentityChange(false);
}

/** On an identity change, switch to the calmest trajectory of that speaker
 *  (how much a person moves is mostly a property of the speaker). */
function onIdentityChange(userDriven) {
  const id = $("#genIdentity").value || "";
  if (id === state.identityId && !userDriven) return;
  state.identityId = id;
  state.trajAllSpeakers = false;
  const cur = $("#genTrajectory").value;
  const own = ownTrajectories();
  const stillValid = cur && own.some((t2) => t2.id === cur);
  if (!stillValid) {
    const pick = quietestOwnTrajectory();
    if (pick) { $("#genTrajectory").value = pick; if (state.draft) state.draft.trajectoryId = pick; }
  }
  emit("identity", state);
  renderTrajPicker();
}

function setTrajectory(id) {
  trajTouched = true;
  $("#genTrajectory").value = id || "";
  if (state.draft) { state.draft.trajectoryId = id || ""; emit("draft", state); }
  renderTrajPicker();
  showTab("generate");
}

let trajTouched = false;

function applyDefaultTrajectory() {
  // Start on a calm, near-static path instead of "auto" (which inherited whatever the
  // seed clip was doing). The user can still pick anything, and their choice sticks.
  if (trajTouched || $("#genTrajectory").value) return;
  const id = state.config.default_trajectory || (state.assetMeta.trajectories || {}).default || "";
  if (!id || !asset("trajectories", id)) return;
  $("#genTrajectory").value = id;
  if (state.draft) state.draft.trajectoryId = id;
}

function renderTrajPicker() {
  const id = $("#genTrajectory").value;
  const x = id ? asset("trajectories", id) : null;
  const img = $("#genTrajThumb");
  if (x && x.media_url) { img.src = x.media_url; img.style.visibility = "visible"; } else { img.removeAttribute("src"); img.style.visibility = "hidden"; }
  $("#genTrajName").textContent = x ? displayName(x) : t("auto");
  const f = x ? x.features || {} : {};
  $("#genTrajMeta").textContent = x
    ? [x.category ? catLabel(x.category, "trajectories") : "", x.frequency ? t(`tfreq_${x.frequency}`) : "",
       x.duration_sec ? `${Number(x.duration_sec).toFixed(0)} s` : "", f.path_length !== undefined ? `${Number(f.path_length).toFixed(1)} m` : ""].filter(Boolean).join(" · ")
    : "";
  const pick = $("#genTrajPicker");
  if (pick) pick.classList.toggle("is-default", !!(x && x.is_default));

  // explain a borrowed trajectory: whose walking style it is and the height difference
  const host = $("#genTrajWarn") || (() => {
    const n = el("div", { id: "genTrajWarn", class: "callout warn", style: { marginTop: "6px" } });
    pick && pick.parentNode && pick.parentNode.insertBefore(n, pick.nextSibling);
    return n;
  })();
  const key = identitySpeakerKey();
  if (x && key && x.speaker_key !== key) {
    const ident = asset("identities", state.identityId);
    const src = (state.assets.identities || []).find((i) => i.speaker_key === x.speaker_key);
    const hs = src && src.stature_m ? `${Number(src.stature_m).toFixed(2)} m` : "?";
    const hi = ident && ident.stature_m ? `${Number(ident.stature_m).toFixed(2)} m` : "?";
    // the walking style matters more than the height, so it comes first
    const own = ownTrajectories();
    const mean = own.length ? own.reduce((a, c) => a + (c.amplitude_score ?? 0), 0) / own.length : null;
    const band = (v) => (v === null ? "" : v < -1 ? "anchored" : v < -0.3 ? "subtle" : v < 0.4 ? "moderate" : v < 1.1 ? "active" : "wide");
    host.textContent = t("traj_mismatch", {
      a: x.speaker || "?", ca: catLabel(x.category || "moderate", "trajectories"), ha: hs,
      b: displayName(ident), cb: catLabel(band(mean), "trajectories"), hb: hi,
    });
    host.classList.remove("hidden");
  } else {
    host.textContent = "";
    host.classList.add("hidden");
  }
}

async function submitGenerate(e) {
  e.preventDefault();
  const msg = $("#genMsg");
  const btn = $("#btnGenerate");
  const file = $("#audioInput").files && $("#audioInput").files[0];
  const reuse = $("#reuseAudio").checked && state.job && state.job.audio_url;
  if (!file && !reuse) { msg.className = "form-msg err"; msg.textContent = t("no_audio"); return; }
  const fd = new FormData();
  fd.set("action", "generate");
  if (file) fd.set("audio", file); else fd.set("audio_job_id", state.job.job_id);
  fd.set("identity", $("#genIdentity").value);
  fd.set("trajectory", $("#genTrajectory").value);
  fd.set("title", $("#genTitle").value);
  fd.set("guidance_scale", $("#genGuidance").value);
  fd.set("render", $("#genRender").checked ? "1" : "0");
  fd.set("skip_asr", $("#genSkipAsr").checked ? "1" : "0");
  const draftTraj = state.mode === "draft" ? state.pending.traj : [];
  fd.set("traj_script", JSON.stringify(draftTraj.map(cleanSeg)));
  btn.disabled = true;
  msg.className = "form-msg"; msg.textContent = "…";
  try {
    const status = await api.createJob(fd);
    msg.className = "form-msg ok"; msg.textContent = `${t("queued")} · ${status.job_id}`;
    toast(`${t("queued")}: ${status.job_id}`, "ok");
    // the draft's moves now live in the job request
    state.draft = null; $("#audioInput").value = ""; $("#audioDrop").classList.remove("has"); $("#audioName").textContent = ""; $("#genTitle").value = "";
    clearPending();
    emit("draft", state);
    await refreshJobs();
    await selectJob(status.job_id);
    showTab("info");
  } catch (err) {
    msg.className = "form-msg err"; msg.textContent = err.message;
  } finally { btn.disabled = false; }
}

function cleanSeg(s) {
  const out = { id: s.id, start: s.start, end: s.end, type: s.type, amount: s.amount, mode: s.mode, hold: s.hold };
  if (s.steps !== undefined && s.steps !== null && s.steps !== "") out.steps = Number(s.steps);
  return out;
}

// ------------------------------------------------------------------ edit page
function renderEdit() {
  const host = $("#inspEditBody");
  host.innerHTML = "";
  const src = timelineSource();
  if (src.kind === "none") {
    host.append(el("div", { class: "empty" }, [icon("layers"), el("div", { text: t("no_job_selected") })]));
    return;
  }
  const item = selectedItem();
  if (item && state.selection.type === "kp") host.append(keyposeEditor(item));
  else if (item && state.selection.type === "tr") host.append(trajEditor(item));
  host.append(pendingSection());
  if (src.kind === "job") host.append(aiSection(), optionsSection(), committedSection());
  else host.append(el("div", { class: "callout info", text: `${t("draft_hint")} ${t("generated_with_job")}.` }));
}

function field(label, control, hint) {
  return el("div", { class: "field" }, [el("label", { text: label }), control, hint ? el("div", { class: "hint", text: hint }) : null]);
}

function slider(value, min, max, step, onChange, fmt = (v) => v) {
  const range = el("input", { type: "range", min, max, step, value });
  const num = el("input", { type: "number", class: "input sm", min, max, step, value: fmt(value) });
  range.addEventListener("input", () => { num.value = fmt(Number(range.value)); onChange(Number(range.value), false); });
  range.addEventListener("change", () => onChange(Number(range.value), true));
  num.addEventListener("change", () => { const v = clamp(Number(num.value), min, max); range.value = v; num.value = fmt(v); onChange(v, true); });
  return el("div", { class: "slider" }, [range, num]);
}

function keyposeEditor(k) {
  const kp = asset("keyposes", k.keypose_id);
  const src = timelineSource();
  const card = el("div", { class: "insp-card kp" });
  card.append(el("div", { class: "insp-card-head" }, [
    kp && kp.media_url ? el("img", { src: kp.media_url, alt: "" }) : null,
    el("div", { class: "t" }, [el("div", { class: "name", text: kp ? displayName(kp) : k.keypose_id }), el("div", { class: "meta", text: kp ? `${catLabel(kp.category, "keyposes")} · ${kp.speaker || ""}` : "" })]),
    el("button", { class: "btn icon xs ghost", title: t("remove"), onclick: () => removePending("kp", k.id) }, [icon("trash")]),
  ]));
  // frame
  const frameIn = el("input", { type: "number", class: "input sm", min: 0, max: src.totalFrames - 1, value: k.frame });
  frameIn.addEventListener("change", () => { updatePending("kp", k.id, { frame: clamp(Number(frameIn.value), 0, src.totalFrames - 1) }); seek(Number(frameIn.value), "ui"); });
  card.append(field(t("frame"), el("div", { class: "grid2" }, [
    frameIn,
    el("button", { class: "btn sm ghost", onclick: () => { updatePending("kp", k.id, { frame: state.playhead }); } }, [icon("step-fwd"), t("set_to_playhead")]),
  ]), `${fmtTime(k.frame, src.fps)}`));
  const libSel = state.librarySel.keyposes;
  if (libSel && libSel !== k.keypose_id) {
    const alt = asset("keyposes", libSel);
    card.append(el("button", { class: "btn sm ghost", style: { justifyContent: "flex-start" }, onclick: () => updatePending("kp", k.id, { keypose_id: libSel }) }, [icon("pose"), `${t("replace_with_selected")}: ${alt ? displayName(alt) : libSel}`]));
  }
  const word = nearestWord(k.frame);
  if (word) {
    const sent = sentenceAt(word.sf);
    card.append(el("div", { class: "word-here", title: sent ? sent.text : "" }, [el("span", { text: `${t("word_here")}:` }), el("b", { text: word.text }), el("span", { class: "muted", text: `f${word.sf}–${word.ef - 1}` })]));
  }
  // part
  const partSel = el("select", { class: "select sm" }, (state.config.body_parts || []).map((p) => el("option", { value: p, text: t(`part_${p}`), selected: p === k.part })));
  partSel.addEventListener("change", () => updatePending("kp", k.id, { part: partSel.value }));
  card.append(field(t("part"), partSel));
  card.append(field(t("strength"), slider(k.strength, 0, 1, 0.05, (v, commit) => { k.strength = v; if (commit) updatePending("kp", k.id, { strength: v }); }, (v) => Number(v).toFixed(2))));
  card.append(field(t("sigma"), slider(k.sigma, 1, 30, 0.5, (v, commit) => { k.sigma = v; if (commit) updatePending("kp", k.id, { sigma: v }); else emit("pendingLive", state); }, (v) => Number(v).toFixed(1)), `±2σ ≈ ${(4 * k.sigma / src.fps).toFixed(2)} s`));
  const bands = el("div", { class: "bands" }, (state.config.bands || []).map((b) => {
    const cb = el("input", { type: "checkbox", checked: (k.bands || []).includes(b) });
    cb.addEventListener("change", () => { const set = new Set(k.bands || []); if (cb.checked) set.add(b); else set.delete(b); updatePending("kp", k.id, { bands: (state.config.bands || []).filter((x) => set.has(x)) }); });
    return el("label", {}, [cb, el("span", { text: b.toUpperCase() })]);
  }));
  card.append(field(t("bands"), bands));
  return card;
}

function trajEditor(s) {
  const src = timelineSource();
  const prims = state.config.primitives || {};
  const card = el("div", { class: "insp-card tr" });
  card.append(el("div", { class: "insp-card-head" }, [
    el("div", { class: "pending-item", style: { border: 0, padding: 0, background: "none", cursor: "default" } }, [el("span", { class: "ic tr" }, [icon(primIcon(s.type))])]),
    el("div", { class: "t" }, [el("div", { class: "name", text: segLabel(s) }), el("div", { class: "meta", text: `f${s.start}–${s.end - 1} · ${fmtTime(s.end - s.start, src.fps)}` })]),
    el("button", { class: "btn icon xs ghost", title: t("remove"), onclick: () => removePending("tr", s.id) }, [icon("trash")]),
  ]));
  const order = (state.config.primitive_order || Object.keys(prims)).filter((k) => prims[k]);
  const grid = el("div", { class: "prim-grid" }, order.map((type) => el("button", {
    type: "button", class: type === s.type ? "on" : "",
    onclick: () => {
      const patch = { type };
      const p = prims[type];
      if (p.channel === "dz" || p.channel === "dx") { patch.steps = s.steps !== undefined && s.steps !== null ? s.steps : 2; patch.amount = patch.steps * (state.config.step_length || 0.6); }
      else { patch.steps = undefined; patch.amount = p.default; }
      updatePending("tr", s.id, patch);
      select("tr", s.id);
    },
  }, [icon(primIcon(type)), el("span", { text: t(type === "hold" ? "hold_still" : type) })])));
  card.append(field(t("type"), grid));
  const p = prims[s.type] || {};
  if (p.channel === "dz" || p.channel === "dx") {
    const stepsIn = el("input", { type: "number", class: "input sm", min: 0.5, max: 10, step: 0.5, value: s.steps !== undefined && s.steps !== null ? s.steps : (s.amount / (state.config.step_length || 0.6)).toFixed(1) });
    const mIn = el("input", { type: "number", class: "input sm", min: 0.05, max: 6, step: 0.05, value: Number(s.amount).toFixed(2) });
    stepsIn.addEventListener("change", () => { const st = Number(stepsIn.value); updatePending("tr", s.id, { steps: st, amount: st * (state.config.step_length || 0.6) }); });
    mIn.addEventListener("change", () => updatePending("tr", s.id, { steps: undefined, amount: Number(mIn.value) }));
    card.append(field(t("amount"), el("div", { class: "grid2" }, [
      el("div", { class: "slider", style: { gridTemplateColumns: "1fr auto" } }, [stepsIn, el("span", { class: "muted", text: t("steps") })]),
      el("div", { class: "slider", style: { gridTemplateColumns: "1fr auto" } }, [mIn, el("span", { class: "muted", text: "m" })]),
    ]), `1 ${t("steps")} = ${state.config.step_length || 0.6} m`));
  } else if (p.channel === "yaw") {
    card.append(field(`${t("amount")} (${t("degrees")})`, slider(s.amount, 5, 180, 5, (v, commit) => { s.amount = v; if (commit) updatePending("tr", s.id, { amount: v }); else emit("pendingLive", state); }, (v) => Math.round(v))));
  } else if (p.channel === "y") {
    card.append(field(`${t("amount")} (m)`, slider(s.amount, 0.02, 0.5, 0.01, (v, commit) => { s.amount = v; if (commit) updatePending("tr", s.id, { amount: v }); else emit("pendingLive", state); }, (v) => Number(v).toFixed(2)), s.type === "crouch" ? "0.08–0.15 m ≈ a gentle dip; deeper values leave the training distribution." : ""));
    card.append(field(t("hold"), slider(s.hold ?? 0.4, 0, 0.9, 0.05, (v, commit) => { s.hold = v; if (commit) updatePending("tr", s.id, { hold: v }); }, (v) => Number(v).toFixed(2))));
  }
  // range
  const startIn = el("input", { type: "number", class: "input sm", min: 0, max: src.totalFrames - 1, value: s.start });
  const endIn = el("input", { type: "number", class: "input sm", min: 1, max: src.totalFrames, value: s.end });
  const durIn = el("input", { type: "number", class: "input sm", min: 0.1, step: 0.1, value: ((s.end - s.start) / src.fps).toFixed(2) });
  startIn.addEventListener("change", () => { const st = clamp(Number(startIn.value), 0, s.end - 4); updatePending("tr", s.id, { start: st }); });
  endIn.addEventListener("change", () => { const en = clamp(Number(endIn.value), s.start + 4, src.totalFrames); updatePending("tr", s.id, { end: en }); });
  durIn.addEventListener("change", () => { const en = clamp(s.start + Math.round(Number(durIn.value) * src.fps), s.start + 4, src.totalFrames); updatePending("tr", s.id, { end: en }); });
  card.append(el("div", { class: "grid2" }, [field(`${t("start")} (${t("frames")})`, startIn), field(`${t("end")} (${t("frames")})`, endIn)]));
  card.append(field(`${t("duration")} (s)`, durIn));
  if (p.channel !== "y" && p.channel !== null && p.channel !== undefined) {
    const modeSeg = el("div", { class: "seg" }, ["replace", "add"].map((m) => el("button", { type: "button", class: s.mode === m ? "on" : "", onclick: () => updatePending("tr", s.id, { mode: m }) }, [t(m)])));
    card.append(field(t("mode"), modeSeg));
  }
  return card;
}

function pendingSection() {
  const sec = el("div", { class: "insp-section" });
  sec.append(el("h3", { text: `${t("pending_edits")} (${pendingCount()})` }));
  if (!pendingCount()) { sec.append(el("div", { class: "hint", text: t("no_pending") })); return sec; }
  const list = el("div", { class: "pending-list" });
  const sel = state.selection;
  state.pending.keyposes.forEach((k) => {
    const kp = asset("keyposes", k.keypose_id);
    list.append(el("div", { class: `pending-item ${sel && sel.id === k.id ? "on" : ""}`, title: k.reason || "", onclick: () => { select("kp", k.id); seek(k.frame, "ui"); } }, [
      el("span", { class: "ic kp" }, [icon("pose")]),
      el("span", { class: "t", text: `${kp ? displayName(kp) : k.keypose_id} · ${t(`part_${k.part}`)}` }),
      k.by === "ai" ? el("span", { class: "ai-badge", title: t("ai_by"), text: "AI" }) : null,
      el("span", { class: "f", text: `f${k.frame}` }),
    ]));
  });
  state.pending.traj.forEach((s) => {
    list.append(el("div", { class: `pending-item ${sel && sel.id === s.id ? "on" : ""}`, onclick: () => { select("tr", s.id); seek(s.start, "ui"); } }, [
      el("span", { class: "ic tr" }, [icon(primIcon(s.type))]), el("span", { class: "t", text: segLabel(s) }), el("span", { class: "f", text: `f${s.start}–${s.end - 1}` }),
    ]));
  });
  sec.append(list);
  return sec;
}

// ------------------------------------------------------------------ AI edit assistant
// proposals always go to the pending edits and are never applied directly
const agentUI = {
  instruction: storage("studio.agent.instruction") || "",
  max: null, gap: null, busy: false, last: null,
  startedAt: 0, error: "", statusNode: null, tick: null,
};

function agentDefaults() {
  return (state.config.agent && state.config.agent.defaults) || { max_edits: 12, min_gap_frames: 45 };
}

function aiSection() {
  const job = state.job;
  const d = agentDefaults();
  if (agentUI.max === null) agentUI.max = d.max_edits;
  if (agentUI.gap === null) agentUI.gap = d.min_gap_frames;
  const sec = el("div", { class: "insp-section agent" });
  const head = el("h3", {}, [
    el("span", { text: t("ai_director") }),
    el("button", { class: "btn icon xs ghost", title: t("ai_settings"), onclick: openAgentSettings }, [icon("cog")]),
  ]);
  sec.append(head);
  sec.append(el("div", { class: "hint", text: t("ai_hint") }));

  const box = el("textarea", { class: "input", rows: 2, placeholder: t("ai_instruction_ph") }, [agentUI.instruction]);
  box.addEventListener("input", () => { agentUI.instruction = box.value; storage("studio.agent.instruction", box.value); });
  sec.append(el("div", { class: "field", style: { marginTop: "8px" } }, [el("label", { text: t("ai_instruction") }), box]));

  const src0 = timelineSource();
  const fps = src0.fps || 30;
  const secs = (src0.totalFrames || 0) / fps;
  const maxIn = el("input", { type: "number", class: "input sm", min: 1, max: 40, step: 1, value: agentUI.max });
  const gapIn = el("input", { type: "number", class: "input sm", min: 0, max: 600, step: 5, value: agentUI.gap });
  const maxHint = el("div", { class: "hint" });
  const gapHint = el("div", { class: "hint" });
  // the hint follows the input live and converts frames to seconds
  const syncHints = () => {
    gapHint.textContent = t("ai_gap_hint", { n: agentUI.gap, s: (agentUI.gap / fps).toFixed(1) });
    maxHint.textContent = secs > 0 ? t("ai_max_hint", { d: secs.toFixed(0), e: (secs / Math.max(1, agentUI.max)).toFixed(1) }) : "";
  };
  maxIn.addEventListener("input", () => { agentUI.max = clamp(Number(maxIn.value) || 1, 1, 40); syncHints(); });
  gapIn.addEventListener("input", () => { agentUI.gap = clamp(Number(gapIn.value) || 0, 0, 600); syncHints(); });
  maxIn.addEventListener("change", () => { maxIn.value = agentUI.max; });
  gapIn.addEventListener("change", () => { gapIn.value = agentUI.gap; });
  syncHints();
  sec.append(el("div", { class: "grid2" }, [
    el("div", { class: "field" }, [el("label", { text: t("ai_max") }), maxIn, maxHint]),
    el("div", { class: "field" }, [el("label", { text: `${t("ai_gap")} (${t("frames")})` }), gapIn, gapHint]),
  ]));

  const ready = job && job.state === "succeeded";
  const btn = el("button", { class: "btn primary block", disabled: !ready || agentUI.busy, onclick: () => runAgent() },
    [icon(agentUI.busy ? "loop" : "pose"), agentUI.busy ? t("ai_thinking") : t("ai_suggest")]);
  sec.append(el("div", { style: { marginTop: "8px" } }, [btn]));

  // show that the request is running (it can take minutes): label and elapsed time
  if (agentUI.busy) {
    const el0 = el("div", { class: "agent-status" }, [el("span", { class: "dot" }), el("span", { class: "txt" })]);
    agentUI.statusNode = el0.querySelector(".txt");
    const paint = () => {
      if (!agentUI.statusNode) return;
      const s = Math.round((Date.now() - agentUI.startedAt) / 1000);
      agentUI.statusNode.textContent = `${t("ai_thinking")} ${s}s · ${t("ai_endpoint")} ${(state.config.agent && state.config.agent.defaults.model) || ""}`;
    };
    paint();
    sec.append(el0);
  }
  // keep the failure reason in the panel, not only in a toast
  if (agentUI.error) {
    sec.append(el("div", { class: "callout err", style: { marginTop: "8px", whiteSpace: "pre-wrap" }, text: agentUI.error }));
  }
  if (job && !job.words_available) sec.append(el("div", { class: "callout warn", style: { marginTop: "8px" }, text: t("ai_needs_words") }));

  const last = agentUI.last;
  if (last) {
    const bits = [last.model, last.usage && last.usage.total_tokens ? `${last.usage.total_tokens} tok` : "",
      last.context && last.context.words ? `${last.context.words} ${t("words").toLowerCase()}` : ""].filter(Boolean).join(" · ");
    sec.append(el("div", { class: "agent-result" }, [
      last.summary ? el("div", { class: "sum", text: last.summary }) : null,
      bits ? el("div", { class: "muted mono", text: bits }) : null,
      (last.notes || []).length ? el("details", {}, [
        el("summary", { text: `${t("ai_dropped")} (${last.notes.length})` }),
        el("div", { class: "hint" }, last.notes.map((n) => el("div", { text: n }))),
      ]) : null,
    ]));
  }
  return sec;
}

async function runAgent() {
  const job = state.job;
  if (!job || job.state !== "succeeded") { toast(t("ai_needs_job"), "warn"); return; }
  agentUI.busy = true;
  agentUI.error = "";
  agentUI.startedAt = Date.now();
  renderEdit();                                  // draw the running state before the request
  clearInterval(agentUI.tick);
  agentUI.tick = setInterval(() => {
    if (!agentUI.statusNode) return;
    const s = Math.round((Date.now() - agentUI.startedAt) / 1000);
    agentUI.statusNode.textContent = `${t("ai_thinking")} ${s}s`;
  }, 1000);
  try {
    const res = await api.agentSuggest(job.job_id, {
      instruction: agentUI.instruction, max_edits: agentUI.max, min_gap_frames: agentUI.gap,
    });
    agentUI.last = res;
    const edits = res.edits || [];
    // part / strength are fixed by the server (full_body / 1.0)
    edits.forEach((e) => addKeypose(e.keypose_id, e.frame, {
      part: e.part, strength: e.strength, sigma: e.sigma, by: "ai", reason: e.reason || "", label: e.reason || e.label || "",
    }));
    select(null);
    toast(edits.length ? t("ai_added", { n: edits.length }) : t("ai_none"), edits.length ? "ok" : "warn");
  } catch (err) {
    agentUI.last = null;
    agentUI.error = err.message;
    toast(err.message, "err", 9000);
    if (/api key|base url/i.test(err.message)) openAgentSettings();
  } finally {
    clearInterval(agentUI.tick);
    agentUI.tick = null;
    agentUI.statusNode = null;
    agentUI.busy = false;
    renderEdit();
  }
}

function openAgentSettings() {
  modal(async (card, close) => {
    card.append(el("h2", { text: `${t("ai_director")} · ${t("ai_settings")}` }));
    const body = el("div", { class: "form" }, [el("div", { class: "hint", text: t("loading") })]);
    card.append(body);
    let cfg = {};
    try { cfg = await api.agentConfig(); } catch (err) { body.innerHTML = ""; body.append(el("div", { class: "callout err", text: err.message })); return; }
    body.innerHTML = "";
    const base = el("input", { type: "text", class: "input", value: cfg.base_url || "", spellcheck: "false" });
    const model = el("input", { type: "text", class: "input", value: cfg.model || "", spellcheck: "false" });
    const proto = el("select", { class: "select" }, ["auto", "openai", "anthropic"].map((v) => el("option", { value: v, text: v, selected: v === (cfg.protocol || "auto") })));
    const key = el("input", { type: "password", class: "input", placeholder: cfg.has_key ? `${cfg.key_hint} — ${t("ai_key_kept")}` : "sk-…", autocomplete: "off" });
    const ep = el("div", { class: "hint mono", text: `${t("ai_endpoint")}: ${cfg.endpoint || "—"}` });
    const sync = () => { ep.textContent = `${t("ai_endpoint")}: ${guessEndpoint(base.value, proto.value)}`; };
    base.addEventListener("input", sync); proto.addEventListener("change", sync);
    body.append(
      field(t("ai_base_url"), base, "OpenAI-compatible: https://<host>/v1 · Anthropic-compatible: https://<host>"),
      field(t("ai_model"), model, "model name as the endpoint expects it"),
      field(t("ai_protocol"), proto),
      field(t("ai_key"), key),
      ep,
    );
    const msg = el("div", { class: "form-msg" });
    card.append(msg, el("div", { class: "actions" }, [
      el("button", { class: "btn", onclick: close }, [t("cancel")]),
      el("button", { class: "btn primary", onclick: async (e) => {
        const b = e.currentTarget; b.disabled = true; msg.className = "form-msg"; msg.textContent = "…";
        try {
          await api.saveAgentConfig({ base_url: base.value, model: model.value, protocol: proto.value, api_key: key.value });
          toast(t("ai_saved"), "ok"); close();
        } catch (err) { msg.className = "form-msg err"; msg.textContent = err.message; b.disabled = false; }
      } }, [t("save")]),
    ]));
  });
}

function guessEndpoint(base, proto) {
  const b = String(base || "").replace(/\/+$/, "");
  if (!b) return "—";
  const p = proto === "auto" ? (b.endsWith("/v1") ? "openai" : "anthropic") : proto;
  if (p === "openai") return b + (b.endsWith("/v1") ? "/chat/completions" : "/v1/chat/completions");
  return b + (b.endsWith("/v1") ? "/messages" : "/v1/messages");
}

function optionsSection() {
  const sec = el("div", { class: "insp-section" });
  sec.append(el("h3", { text: t("apply_edits") }));
  const opts = state.editOptions;
  sec.append(field(t("context_windows"), slider(opts.regen_windows, 1, 8, 1, (v) => { opts.regen_windows = v; }), t("context_hint")));
  const render = el("input", { type: "checkbox", checked: opts.render });
  render.addEventListener("change", () => { opts.render = render.checked; });
  sec.append(el("label", { class: "check" }, [render, el("span", { text: t("render_previews") })]));
  const canApply = pendingCount() && state.job && state.job.state === "succeeded";
  sec.append(el("div", { style: { display: "flex", gap: "6px", marginTop: "8px" } }, [
    el("button", { class: "btn primary", style: { flex: 1 }, disabled: !canApply, onclick: () => applyEdits() }, [icon("check"), t("apply_edits")]),
    el("button", { class: "btn ghost", disabled: !pendingCount(), onclick: async () => { if (await confirm(t("confirm_discard"), { ok: t("discard"), cancel: t("cancel"), danger: true })) clearPending(); } }, [t("discard")]),
  ]));
  if (state.job && state.job.state !== "succeeded") sec.append(el("div", { class: "callout warn", style: { marginTop: "8px" }, text: `${t("state")}: ${state.job.state} — ${state.job.message || ""}` }));
  return sec;
}

function committedSection() {
  const job = state.job;
  const sec = el("div", { class: "insp-section" });
  const n = (job.keyposes || []).length + (job.traj_script || []).length;
  sec.append(el("h3", { text: `${t("committed_edits")} (${n})` }));
  if (!n) {
    sec.append(el("div", { class: "hint", text: "—" }));
  } else {
    const list = el("div", { class: "pending-list" });
    (job.keyposes || []).forEach((k) => {
      const kp = asset("keyposes", k.keypose_id);
      list.append(el("div", { class: "pending-item", style: { opacity: 0.75 }, onclick: () => seek(k.frame, "ui") }, [el("span", { class: "ic kp" }, [icon("pose")]), el("span", { class: "t", text: `${kp ? displayName(kp) : k.keypose_id} · ${t(`part_${k.part}`)}` }), el("span", { class: "f", text: `f${k.frame}` })]));
    });
    (job.traj_script || []).forEach((s) => {
      list.append(el("div", { class: "pending-item", style: { opacity: 0.75 }, onclick: () => seek(s.start, "ui") }, [el("span", { class: "ic tr" }, [icon(primIcon(s.type))]), el("span", { class: "t", text: s.description || segLabel(s) }), el("span", { class: "f", text: `f${s.start}–${s.end - 1}` })]));
    });
    sec.append(list);
  }

  // edits of ancestor jobs: already in this motion (every edit builds on its parent's output)
  // but not in this job's request
  const ik = job.inherited_keyposes || [];
  const it = job.inherited_traj || [];
  if (ik.length || it.length) {
    sec.append(el("h3", { style: { marginTop: "12px" }, text: `${t("inherited_edits")} (${ik.length + it.length})` }));
    sec.append(el("div", { class: "hint", style: { marginBottom: "6px" }, text: t("inherited_hint") }));
    const ilist = el("div", { class: "pending-list" });
    const srcLink = (jid) => el("a", { class: "f", href: "#", text: shortId(jid), title: `${t("from_job")}: ${jid}`, onclick: (e) => { e.preventDefault(); e.stopPropagation(); selectJob(jid); } });
    ik.forEach((k) => {
      const kp = asset("keyposes", k.keypose_id);
      ilist.append(el("div", { class: "pending-item", style: { opacity: 0.5 }, onclick: () => seek(k.frame, "ui") }, [
        el("span", { class: "ic kp" }, [icon("pose")]),
        el("span", { class: "t", text: `${kp ? displayName(kp) : k.keypose_id} · ${t(`part_${k.part}`)} · f${k.frame}` }),
        srcLink(k.from_job),
      ]));
    });
    it.forEach((s) => {
      ilist.append(el("div", { class: "pending-item", style: { opacity: 0.5 }, onclick: () => seek(s.start, "ui") }, [
        el("span", { class: "ic tr" }, [icon(primIcon(s.type))]),
        el("span", { class: "t", text: `${s.description || segLabel(s)} · f${s.start}–${s.end - 1}` }),
        srcLink(s.from_job),
      ]));
    });
    sec.append(ilist);
  }

  if (job.parent_id) sec.append(el("div", { class: "hint", style: { marginTop: "6px" } }, [`${t("parent")}: `, el("a", { href: "#", text: shortId(job.parent_id), onclick: (e) => { e.preventDefault(); selectJob(job.parent_id); } })]));
  return sec;
}

export async function applyEdits() {
  if (state.mode !== "job" || !state.job) { toast(t("draft_hint"), "warn"); return; }
  if (!pendingCount()) { toast(t("nothing_to_apply"), "warn"); return; }
  if (state.job.state !== "succeeded") { toast(`${t("state")}: ${state.job.state}`, "warn"); return; }
  const body = {
    keyposes: state.pending.keyposes.map((k) => ({ id: k.id, keypose_id: k.keypose_id, frame: k.frame, part: k.part, strength: k.strength, sigma: k.sigma, bands: k.bands, label: k.label || k.reason || "" })),
    traj_script: state.pending.traj.map(cleanSeg),
    regen_windows: state.editOptions.regen_windows,
    render: state.editOptions.render,
    title: state.job.title ? `${state.job.title} · edit` : "",
  };
  try {
    const status = await api.editJob(state.job.job_id, body);
    toast(`${t("applying")}: ${status.job_id}`, "ok");
    clearPending();
    await refreshJobs();
    await selectJob(status.job_id);
    showTab("info");
  } catch (err) { toast(err.message, "err", 6000); }
}

function highlightTranscript() {
  const f = state.playhead;
  $$("#inspInfoBody .transcript .sent").forEach((s) => s.classList.toggle("cur", Number(s.dataset.sf) <= f && f < Number(s.dataset.ef)));
}

// ------------------------------------------------------------------ info page
function renderInfo() {
  const host = $("#inspInfoBody");
  if (host.contains(document.activeElement) && document.activeElement.tagName === "TEXTAREA") return; // user is typing notes
  host.innerHTML = "";
  const job = state.job;
  if (!job) { host.append(el("div", { class: "empty" }, [icon("layers"), el("div", { text: t("no_job_selected") })])); return; }
  const dl = el("dl", { class: "kv" });
  const row = (k, v, mono = false) => { if (v === undefined || v === null || v === "") return; dl.append(el("dt", { text: k }), el("dd", { class: mono ? "mono" : "", text: String(v) })); };
  row("ID", job.job_id, true);
  row(t("state"), `${job.state}${job.phase ? ` · ${job.phase}` : ""}`);
  row(t("created"), fmtDate(job.created_at));
  row(t("elapsed"), fmtDuration(job.elapsed_seconds));
  row(t("audio"), job.audio_name);
  row(t("identity"), job.identity || t("auto"));
  row(t("base_trajectory"), job.trajectory || t("auto"));
  row(t("frames"), job.total_frames ? `${job.total_frames} (${fmtTime(job.total_frames, job.fps)})` : "");
  row(t("chunks"), (job.chunks || []).length || "");
  const ck = (job.manifest && job.manifest.checkpoints) || {};
  if (ck.body) row(t("checkpoints"), String(ck.body).split("/").slice(-2).join("/"), true);
  host.append(el("div", { class: "insp-section" }, [el("h3", { text: t("job") }), dl]));
  if (job.message && ["failed", "cancelled"].includes(job.state)) host.append(el("div", { class: "callout err", text: job.message }));
  if ((job.warnings || []).length) host.append(el("div", { class: "callout warn", text: job.warnings.join("\n") }));
  const notes = el("textarea", { class: "input", rows: 2, placeholder: t("notes") }, [job.notes || ""]);
  notes.addEventListener("change", async () => { try { await api.patchJob(job.job_id, { notes: notes.value }); toast(t("saved"), "ok", 1000); } catch (err) { toast(err.message, "err"); } });
  host.append(el("div", { class: "insp-section" }, [el("h3", { text: t("notes") }), notes]));
  if (state.words && (state.words.sentences || []).length) {
    host.append(el("div", { class: "insp-section" }, [el("h3", { text: `${t("transcript")} · ${state.words.words.length} ${t("words")}` }),
      el("div", { class: "transcript" }, state.words.sentences.map((s) => el("span", { class: "sent", dataset: { sf: s.sf, ef: s.ef }, title: `${fmtTime(s.sf, job.fps)} · f${s.sf}`, text: `${s.text} `, onclick: () => seek(s.sf, "ui") })))]));
    highlightTranscript();
  } else if (job.transcripts && Array.isArray(job.transcripts.chunks || job.transcripts)) {
    const rows = job.transcripts.chunks || job.transcripts;
    host.append(el("div", { class: "insp-section" }, [el("h3", { text: t("transcript") }), el("div", { class: "transcript" }, rows.map((r, i) => el("div", {}, [el("b", { text: `#${i + 1}` }), typeof r === "string" ? r : (r.text || r.transcript || JSON.stringify(r))])))]));
  }
  if (!job.words_available && job.audio_url) {
    host.append(el("div", { class: "insp-section" }, [
      el("div", { class: "hint", style: { marginBottom: "6px" }, text: t("no_words_hint") }),
      el("button", { class: "btn sm", disabled: !!job.transcribe_pending, onclick: requestTranscribe }, [icon("cc"), job.transcribe_pending ? t("transcribing") : t("transcribe_words")]),
    ]));
  }
  if ((job.lineage || []).length) {
    host.append(el("div", { class: "insp-section" }, [el("h3", { text: t("lineage") }), el("div", { class: "link-list" }, job.lineage.map((id) => el("a", { href: "#", onclick: (e) => { e.preventDefault(); selectJob(id); } }, [icon("link"), shortId(id)])))]));
  }
  const arts = (job.artifacts || []).filter((a) => a.kind !== "image");
  if (arts.length) {
    host.append(el("div", { class: "insp-section" }, [el("h3", { text: `${t("artifacts")} (${arts.length})` }), el("div", { class: "link-list" }, arts.map((a) => el("a", { href: a.url, target: "_blank", title: a.rel }, [icon(a.kind === "video" ? "play" : a.kind === "audio" ? "loop" : "download"), el("span", { text: a.rel.length > 46 ? `…${a.rel.slice(-44)}` : a.rel }), el("span", { class: "sz", text: fmtBytes(a.size) })])))]));
  }
  const acts = el("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap" } });
  if (["queued", "waiting", "running", "starting"].includes(job.state)) acts.append(el("button", { class: "btn sm danger", onclick: async () => { try { await api.cancelJob(job.job_id); toast(t("cancelled"), "warn"); } catch (err) { toast(err.message, "err"); } } }, [icon("cancel"), t("cancel_job")]));
  if (["succeeded", "failed", "cancelled"].includes(job.state)) acts.append(el("button", { class: "btn sm ghost", onclick: async () => { if (await confirm(t("confirm_delete"), { ok: t("delete"), cancel: t("cancel"), danger: true })) { try { await api.deleteJob(job.job_id); toast(t("deleted"), "ok"); emit("jobDeleted", job.job_id); } catch (err) { toast(err.message, "err"); } } } }, [icon("trash"), t("delete")]));
  host.append(el("div", { class: "insp-section" }, [acts]));
}
