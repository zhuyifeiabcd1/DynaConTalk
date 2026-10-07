// Asset library: keyposes / trajectories / identities with search, filters, inline metadata editing.
import { $, $$, el, icon, toast, debounce, hashColor } from "./util.js";
import { t, onLang } from "./i18n.js";
import { api } from "./api.js";
import { state, emit, on, set, addKeypose, timelineSource, identitySpeakerKey, ownTrajectories } from "./store.js";

const grid = () => $("#libGrid");
let filters = { q: "", speaker: "", category: "", frequency: "" };
let detailAsset = null;

export async function loadAssets() {
  const kinds = ["keyposes", "trajectories", "identities"];
  const results = await Promise.allSettled(kinds.map((k) => api.assets(k)));
  results.forEach((res, i) => {
    const kind = kinds[i];
    if (res.status === "fulfilled") {
      state.assets[kind] = res.value.items || [];
      if (kind === "keyposes" || kind === "trajectories") {
        state.assetMeta[kind] = {
          categories: res.value.categories || [],
          speakers: res.value.speakers || [],
          frequencies: res.value.frequencies || [],
          default: res.value.default || "",
        };
      }
    } else {
      console.warn("assets", kind, res.reason);
    }
  });
  emit("assets", state);
}

export function initLibrary() {
  on("identity", () => { if (state.libraryTab === "trajectories") { render(); renderFilters(); } });
  $$("#libraryTabs button").forEach((btn) => btn.addEventListener("click", () => setTab(btn.dataset.tab)));
  $("#libSearch").addEventListener("input", debounce((e) => { filters.q = e.target.value.trim().toLowerCase(); render(); }, 120));
  $("#libSpeaker").addEventListener("change", (e) => { filters.speaker = e.target.value; render(); });
  on("assets", () => { renderFilters(); render(); renderDetail(); });
  on("job", () => renderDetail());
  on("draft", () => renderDetail());
  onLang(() => { renderFilters(); render(); renderDetail(); });
  renderFilters();
  render();
}

export function setTab(tab) {
  if (!["keyposes", "trajectories", "identities"].includes(tab)) return;
  state.libraryTab = tab;
  $$("#libraryTabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  filters.category = "";
  filters.frequency = "";
  filters.speaker = "";
  $("#libSpeaker").value = "";
  detailAsset = null;
  renderFilters();
  render();
  renderDetail();
}

function items() {
  const tab = state.libraryTab;
  let list = state.assets[tab] || [];
  if (filters.q) {
    const q = filters.q;
    list = list.filter((x) => [x.id, x.name, x.name_zh, x.speaker, x.category, ...(x.tags || []), ...(x.auto_tags || [])].join(" ").toLowerCase().includes(q));
  }
  // trajectories follow the current identity: how much a person moves is mostly a property of the speaker
  if (tab === "trajectories" && !state.trajAllSpeakers) {
    const key = identitySpeakerKey();
    if (key) list = list.filter((x) => x.speaker_key === key);
  }
  if (filters.speaker) list = list.filter((x) => x.speaker === filters.speaker);
  if (filters.category) list = list.filter((x) => (x.category || "other") === filters.category);
  if (filters.frequency) list = list.filter((x) => x.frequency === filters.frequency);
  if (state.libraryTab === "trajectories") {
    // calmest first: the quiet ones are what most presentations want
    list = [...list].sort((a, b) => (a.amplitude_score ?? 0) - (b.amplitude_score ?? 0));
  }
  return list;
}

function renderFilters() {
  const tab = state.libraryTab;
  const spk = $("#libSpeaker");
  const speakers = new Map();
  (state.assets[tab] || []).forEach((x) => { if (x.speaker) speakers.set(x.speaker, (speakers.get(x.speaker) || 0) + 1); });
  const cur = spk.value;
  spk.innerHTML = "";
  spk.append(el("option", { value: "", text: t("all_speakers") }));
  [...speakers.entries()].sort().forEach(([name, n]) => spk.append(el("option", { value: name, text: `${name} (${n})` })));
  spk.value = speakers.has(cur) ? cur : "";
  // with an identity selected the speaker filter is redundant (the list holds one speaker)
  spk.classList.toggle("hidden", tab === "identities"
    || (tab === "trajectories" && !state.trajAllSpeakers && !!identitySpeakerKey()));

  const chips = $("#libChips");
  chips.innerHTML = "";
  if (tab !== "keyposes" && tab !== "trajectories") return;
  if (tab === "trajectories") {
    const key = identitySpeakerKey();
    const ident = state.identityId ? (state.assets.identities || []).find((i) => i.id === state.identityId) : null;
    if (key) {
      const own = ownTrajectories(key).length;
      const total = (state.assets.trajectories || []).length;
      chips.append(el("div", { class: "chip-row bind-row" }, [
        el("span", { class: "chips-label", text: t("bound_to") }),
        el("button", { class: `chip-btn ${state.trajAllSpeakers ? "" : "on"}`, title: t("bound_hint"),
          onclick: () => { state.trajAllSpeakers = false; render(); renderFilters(); } },
          [displayName(ident), el("span", { class: "n", text: own })]),
        el("button", { class: `chip-btn ${state.trajAllSpeakers ? "on" : ""}`, title: t("borrow_hint"),
          onclick: () => { state.trajAllSpeakers = true; render(); renderFilters(); } },
          [t("all_speakers"), el("span", { class: "n", text: total })]),
      ]));
    }
  }
  // with an identity selected, count categories / frequencies of this speaker only, otherwise
  // a chip would promise items that belong to other speakers
  const bindKey2 = tab === "trajectories" && !state.trajAllSpeakers ? identitySpeakerKey() : "";
  const all = bindKey2 ? ownTrajectories(bindKey2) : (state.assets[tab] || []);
  const cats = new Map();
  all.forEach((x) => { const c = x.category || defaultCat(tab); cats.set(c, (cats.get(c) || 0) + 1); });
  const order = catOrder(tab);
  const keys = [...order.filter((c) => cats.has(c)), ...[...cats.keys()].filter((c) => !order.includes(c))];
  const row = el("div", { class: "chip-row" });
  row.append(el("button", { class: `chip-btn ${filters.category ? "" : "on"}`, onclick: () => { filters.category = ""; render(); renderFilters(); } }, [t("all"), el("span", { class: "n", text: all.length })]));
  keys.forEach((c) => row.append(el("button", {
    class: `chip-btn ${filters.category === c ? "on" : ""}`,
    title: tab === "trajectories" ? t(`tcat_${c}_hint`) : "",
    onclick: () => { filters.category = filters.category === c ? "" : c; render(); renderFilters(); },
  }, [catLabel(c, tab), el("span", { class: "n", text: cats.get(c) })])));
  chips.append(row);

  if (tab === "trajectories") {
    const freqs = new Map();
    all.forEach((x) => { if (x.frequency) freqs.set(x.frequency, (freqs.get(x.frequency) || 0) + 1); });
    const fOrder = state.config.trajectory_frequencies || [];
    const fKeys = fOrder.filter((f) => freqs.has(f));
    if (fKeys.length) {
      const sub = el("div", { class: "chip-row" }, [el("span", { class: "chips-label", text: t("frequency") })]);
      fKeys.forEach((f) => sub.append(el("button", {
        class: `chip-btn ${filters.frequency === f ? "on" : ""}`,
        onclick: () => { filters.frequency = filters.frequency === f ? "" : f; render(); renderFilters(); },
      }, [t(`tfreq_${f}`), el("span", { class: "n", text: freqs.get(f) })])));
      chips.append(el("div", { class: "chips sub", style: { padding: 0, width: "100%" } }, [sub]));
    }
  }
}

function catOrder(tab) {
  return tab === "trajectories" ? (state.config.trajectory_categories || []) : (state.config.keypose_categories || []);
}

function defaultCat(tab) { return tab === "trajectories" ? "moderate" : "other"; }

export function catLabel(c, tab = null) {
  const kind = tab || state.libraryTab;
  if (kind === "trajectories") return t(`tcat_${c || "moderate"}`);
  return t(`cat_${c || "other"}`);
}

function render() {
  const host = grid();
  host.innerHTML = "";
  const tab = state.libraryTab;
  host.classList.toggle("list", tab === "identities");
  const list = items();
  if (!list.length) {
    host.append(el("div", { class: "empty" }, [icon("search"), el("div", { text: t("library_empty") })]));
    return;
  }
  const sel = state.librarySel[tab];
  if (tab === "keyposes") {
    // group by category when not filtering by one
    const groups = new Map();
    list.forEach((x) => { const c = x.category || "other"; if (!groups.has(c)) groups.set(c, []); groups.get(c).push(x); });
    const order = state.config.keypose_categories || [];
    const keys = [...order.filter((c) => groups.has(c)), ...[...groups.keys()].filter((c) => !order.includes(c))];
    keys.forEach((c) => {
      if (keys.length > 1) host.append(el("div", { class: "lib-group", text: `${catLabel(c, "keyposes")} · ${groups.get(c).length}` }));
      groups.get(c).forEach((x) => host.append(keyposeCard(x, x.id === sel, keys.length <= 1)));
    });
  } else if (tab === "trajectories") {
    const groups = new Map();
    list.forEach((x) => { const c = x.category || "moderate"; if (!groups.has(c)) groups.set(c, []); groups.get(c).push(x); });
    const order = state.config.trajectory_categories || [];
    const keys = [...order.filter((c) => groups.has(c)), ...[...groups.keys()].filter((c) => !order.includes(c))];
    keys.forEach((c) => {
      if (keys.length > 1) {
        host.append(el("div", { class: "lib-group", title: t(`tcat_${c}_hint`) },
          [el("span", { text: `${catLabel(c, "trajectories")} · ${groups.get(c).length}` }), el("span", { class: "sub", text: t(`tcat_${c}_hint`) })]));
      }
      groups.get(c).forEach((x) => host.append(trajectoryCard(x, x.id === sel)));
    });
  } else {
    list.forEach((x) => host.append(identityCard(x, x.id === sel)));
  }
}

function keyposeCard(x, on, showCat = true) {
  const card = el("div", { class: `card kp ${on ? "on" : ""}`, draggable: "true", dataset: { id: x.id }, title: x.id });
  card.append(
    el("div", { class: "thumb" }, [
      x.media_url ? el("img", { src: x.media_url, alt: x.name, loading: "lazy" }) : null,
      showCat ? el("span", { class: "cat-chip cat", text: catLabel(x.category, "keyposes") }) : null,
      x.time_sec !== undefined && x.time_sec !== null ? el("span", { class: "t", text: `${Number(x.time_sec).toFixed(1)}s` }) : null,
    ]),
    el("div", { class: "body" }, [
      el("div", { class: "name", text: displayName(x) }),
      el("div", { class: "meta", text: [x.speaker, ...(x.tags || []).slice(0, 2)].filter(Boolean).join(" · ") }),
    ]),
    el("div", { class: "quick" }, [
      el("button", { title: `${t("insert_keypose")} (K)`, onclick: (e) => { e.stopPropagation(); insertAtPlayhead(x.id); } }, [icon("plus")]),
    ]),
  );
  card.addEventListener("click", () => selectAsset("keyposes", x.id));
  card.addEventListener("dblclick", () => insertAtPlayhead(x.id));
  card.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/keypose", x.id);
    e.dataTransfer.effectAllowed = "copy";
    selectAsset("keyposes", x.id);
  });
  return card;
}

function trajectoryCard(x, on) {
  const bindKey = identitySpeakerKey();
  const borrowed = bindKey && x.speaker_key !== bindKey;
  const f = x.features || {};
  const m = x.motion || {};
  const cat = x.category || "moderate";
  const card = el("div", { class: `card traj ${on ? "on" : ""} ${borrowed ? "borrowed" : ""}`, dataset: { id: x.id }, title: `${x.id}${m.travel_per_sec !== undefined ? ` · ${m.travel_per_sec} m/s · ${m.moves_per_min} ${t("moves_per_min")}` : ""}${borrowed ? `\n${t("borrow_hint")}` : ""}` });
  card.append(
    el("div", { class: "thumb" }, [
      x.media_url ? el("img", { src: x.media_url, alt: x.name, loading: "lazy" }) : null,
      el("span", { class: `cat-chip cat mo-chip mo-${cat}`, text: x.frequency ? t(`tfreq_${x.frequency}`) : catLabel(cat, "trajectories") }),
      x.is_default && !bindKey ? el("span", { class: "freq-chip def", text: t("default_trajectory") }) : null,
      borrowed ? el("span", { class: "freq-chip borrow", text: t("borrowed") }) : null,
      x.duration_sec ? el("span", { class: "t", text: `${Number(x.duration_sec).toFixed(0)}s` }) : null,
    ]),
    el("div", { class: "body" }, [
      el("div", { class: "name", text: displayName(x) }),
      el("div", { class: "meta", text: [x.speaker, f.path_length !== undefined ? `${Number(f.path_length).toFixed(1)} m` : ""].filter(Boolean).join(" · ") }),
    ]),
    el("div", { class: "quick" }, [
      el("button", { title: t("use_as_base"), onclick: (e) => { e.stopPropagation(); emit("pickTrajectory", x.id); toast(`${t("base_trajectory")}: ${displayName(x)}`, "ok", 1800); } }, [icon("check")]),
    ]),
  );
  card.addEventListener("click", () => selectAsset("trajectories", x.id));
  card.addEventListener("dblclick", () => emit("pickTrajectory", x.id));
  return card;
}

function identityCard(x, on) {
  const card = el("div", { class: `card identity ${on ? "on" : ""}`, dataset: { id: x.id }, title: x.id });
  // body-shape cover (rendered from shape_betas with one camera and crop for everyone, so sizes compare);
  // initials when there is no cover
  const figure = x.media_url
    ? el("img", { class: "figure", src: x.media_url, alt: displayName(x), loading: "lazy" })
    : el("div", { class: "avatar", style: { background: hashColor(x.name) }, text: (x.name || "?").slice(0, 1).toUpperCase() });
  const build = [
    x.speaker_id !== undefined ? `#${x.speaker_id}` : "",
    x.stature_m ? `${Number(x.stature_m).toFixed(2)} m` : "",
    x.shoulder_width_m ? `${t("shoulders")} ${Number(x.shoulder_width_m).toFixed(2)} m` : "",
  ].filter(Boolean).join(" · ");
  card.append(
    figure,
    el("div", { class: "body", style: { padding: 0, minWidth: 0, flex: 1 } }, [
      el("div", { class: "name", text: displayName(x) }),
      el("div", { class: "meta", text: build }),
      el("div", { class: "meta", text: x.representative_sample || "" }),
    ]),
    el("div", { class: "quick", style: { position: "static", display: "flex" } }, [
      el("button", { title: t("set_identity"), onclick: (e) => { e.stopPropagation(); emit("pickIdentity", x.id); toast(`${t("identity")}: ${displayName(x)}`, "ok", 1500); } }, [icon("check")]),
    ]),
  );
  card.addEventListener("click", () => selectAsset("identities", x.id));
  card.addEventListener("dblclick", () => emit("pickIdentity", x.id));
  return card;
}

export function displayName(x) {
  if (!x) return "";
  const zh = document.documentElement.lang.startsWith("zh");
  return (zh && x.name_zh) || x.name || x.id;
}

export function selectAsset(kind, id) {
  state.librarySel[kind] = id;
  $$(`#libGrid .card`).forEach((c) => c.classList.toggle("on", c.dataset.id === id));
  detailAsset = { kind, id };
  renderDetail();
  emit("librarySelect", { kind, id });
}

function insertAtPlayhead(keyposeId) {
  const src = timelineSource();
  if (!src.keyposesEnabled) { toast(t("draft_hint"), "warn"); return; }
  addKeypose(keyposeId, state.playhead);
  toast(t("added_keypose"), "ok", 1400);
}

// ------------------------------------------------------------------ detail / metadata editing
function renderDetail() {
  const host = $("#libFooter");
  host.classList.toggle("on", !!detailAsset);
  host.innerHTML = "";
  if (!detailAsset) return;
  const x = (state.assets[detailAsset.kind] || []).find((a) => a.id === detailAsset.id);
  if (!x) { host.classList.remove("on"); return; }
  const kind = detailAsset.kind;
  const head = el("div", { class: "lib-detail-head" });
  if (x.media_url) head.append(el("img", { src: x.media_url, alt: x.name, class: kind === "trajectories" ? "traj-thumb" : (kind === "identities" ? "body-thumb" : "") }));
  const titleRow = el("div", { class: "lib-detail-title" }, [
    el("div", { class: "name", text: displayName(x) }),
    el("button", { class: "btn icon xs ghost", title: t("rename"), onclick: () => toggleEditor(host, x) }, [icon("edit")]),
  ]);
  const kv = el("div", { class: "lib-detail-kv" });
  const add = (k, v) => { if (v !== undefined && v !== null && v !== "") kv.append(el("b", { text: k }), el("span", { text: String(v) })); };
  add("ID", x.id);
  add(t("speaker"), x.speaker);
  if (kind === "keyposes") { add(t("category"), catLabel(x.category, "keyposes")); add(t("time"), x.time_sec !== undefined ? `${Number(x.time_sec).toFixed(2)} s · f${x.frame}` : ""); }
  if (kind === "trajectories") {
    const f = x.features || {};
    const m = x.motion || {};
    add(t("amplitude"), `${catLabel(x.category, "trajectories")}${x.frequency ? ` · ${t(`tfreq_${x.frequency}`)}` : ""}`);
    add(t("duration"), x.duration_sec ? `${Number(x.duration_sec).toFixed(1)} s · ${x.frames} f` : "");
    add(t("travel_rate"), m.travel_per_sec !== undefined ? `${Number(m.travel_per_sec).toFixed(3)} m/s · ${Number(f.path_length || 0).toFixed(1)} m` : "");
    add(t("moves_row"), m.moves_per_min !== undefined ? `${m.moves_per_min}/min · ${Math.round((m.moving_ratio || 0) * 100)}% ${t("pct_moving")}` : "");
    add(t("turn_rate"), m.turn_per_sec_deg !== undefined ? `${m.turn_per_sec_deg}°/s` : "");
    add("bbox", f.bbox_x !== undefined ? `${Number(f.bbox_x).toFixed(2)} × ${Number(f.bbox_z).toFixed(2)} m` : "");
  }
  if (kind === "identities") {
    add("speaker_id", x.speaker_id);
    add(t("stature"), x.stature_m ? `${Number(x.stature_m).toFixed(2)} m` : "");
    add(t("shoulders"), x.shoulder_width_m ? `${Number(x.shoulder_width_m).toFixed(2)} m` : "");
    add(t("sample"), x.representative_sample);
  }
  head.append(el("div", { style: { minWidth: 0, flex: 1 } }, [titleRow, kv]));
  host.append(head);
  const tags = [...(x.tags || []), ...(kind === "keyposes" ? (x.auto_tags || []).map((s) => `~${s}`) : [])];
  if (tags.length) host.append(el("div", {}, tags.map((s) => el("span", { class: "tag", text: s }))));
  if (x.notes) host.append(el("div", { class: "hint", text: x.notes }));
  const actions = el("div", { style: { display: "flex", gap: "6px" } });
  if (kind === "keyposes") actions.append(el("button", { class: "btn sm accent", onclick: () => insertAtPlayhead(x.id) }, [icon("plus"), t("insert_keypose"), el("kbd", { text: "K" })]));
  if (kind === "trajectories") actions.append(el("button", { class: "btn sm warn", onclick: () => emit("pickTrajectory", x.id) }, [icon("check"), t("use_as_base")]));
  if (kind === "identities") actions.append(el("button", { class: "btn sm", onclick: () => emit("pickIdentity", x.id) }, [icon("check"), t("set_identity")]));
  actions.append(el("button", { class: "btn sm ghost", onclick: () => { detailAsset = null; renderDetail(); } }, [icon("close")]));
  host.append(actions);
}

function toggleEditor(host, x) {
  const existing = host.querySelector(".lib-edit");
  if (existing) { existing.remove(); return; }
  const kind = detailAsset.kind;
  const cats = kind === "trajectories" ? (state.config.trajectory_categories || []) : (state.config.keypose_categories || []);
  const form = el("div", { class: "lib-edit" });
  const name = el("input", { class: "input sm", value: x.name || "", placeholder: "name (EN)" });
  const nameZh = el("input", { class: "input sm", value: x.name_zh || "", placeholder: "Name (Chinese)" });
  const cat = el("select", { class: "select sm" }, cats.map((c) => el("option", { value: c, text: catLabel(c, kind), selected: (x.category || defaultCat(kind)) === c })));
  const tags = el("input", { class: "input sm", value: (x.tags || []).join(", "), placeholder: t("tags") });
  const notes = el("input", { class: "input sm full", value: x.notes || "", placeholder: t("notes") });
  form.append(name, nameZh);
  if (kind === "keyposes" || kind === "trajectories") form.append(cat, tags); else form.append(el("div", { class: "full" }, [tags]));
  form.append(notes);
  form.append(el("div", { class: "full", style: { display: "flex", gap: "6px", justifyContent: "flex-end" } }, [
    el("button", { class: "btn xs ghost", onclick: () => form.remove() }, [t("cancel")]),
    el("button", { class: "btn xs primary", onclick: async () => {
      try {
        const patch = { name: name.value, name_zh: nameZh.value, tags: tags.value, notes: notes.value };
        if (kind === "keyposes" || kind === "trajectories") patch.category = cat.value;
        const updated = await api.patchAsset(kind, x.id, patch);
        const list = state.assets[kind];
        const idx = list.findIndex((a) => a.id === x.id);
        if (idx >= 0) list[idx] = updated;
        emit("assets", state);
        toast(t("saved"), "ok", 1400);
      } catch (err) { toast(err.message, "err"); }
    } }, [t("save")]),
  ]));
  host.append(form);
  name.focus();
}
