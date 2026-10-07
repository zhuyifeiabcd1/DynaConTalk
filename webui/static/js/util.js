// DOM + formatting helpers shared by all modules.
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function icon(name, cls = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  if (cls) svg.setAttribute("class", cls);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const lerp = (a, b, t) => a + (b - a) * t;

export function debounce(fn, ms = 150) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

export function throttle(fn, ms = 60) {
  let last = 0, pending = null;
  return (...args) => {
    const now = performance.now();
    if (now - last >= ms) { last = now; fn(...args); }
    else { clearTimeout(pending); pending = setTimeout(() => { last = performance.now(); fn(...args); }, ms - (now - last)); }
  };
}

export function fmtTime(frames, fps = 30, withFrames = false) {
  const total = Math.max(0, frames) / fps;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  const base = `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
  return withFrames ? `${base} · f${Math.round(frames)}` : base;
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function fmtBytes(n) {
  if (!n && n !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16);
  const pad = (x) => String(x).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function relTime(iso, tr = null) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  const t = tr || ((k, v) => ({ just_now: "just now", min_ago: `${v && v.n} min ago`, h_ago: `${v && v.n} h ago` })[k]);
  if (diff < 60) return t("just_now");
  if (diff < 3600) return t("min_ago", { n: Math.floor(diff / 60) });
  if (diff < 86400) return t("h_ago", { n: Math.floor(diff / 3600) });
  return fmtDate(iso);
}

export function uid(prefix = "id") {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export function hashColor(text) {
  let h = 0;
  for (const ch of String(text)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360} 55% 48%)`;
}

export function storage(key, value) {
  try {
    if (value === undefined) { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; }
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch (_) { return null; }
  return value;
}

export function isTyping() {
  const a = document.activeElement;
  if (!a) return false;
  if (a.isContentEditable || a.tagName === "TEXTAREA" || a.tagName === "SELECT") return true;
  if (a.tagName === "INPUT") return !["checkbox", "radio", "button", "range", "file", "submit"].includes((a.type || "text").toLowerCase());
  return false;
}

// ---- toasts
export function toast(message, tone = "", ms = 3600) {
  const host = $("#toasts");
  const node = el("div", { class: `toast ${tone}` }, [
    el("span", { text: message }),
    el("button", { class: "x", onclick: () => node.remove() }, [icon("close")]),
  ]);
  host.append(node);
  if (ms > 0) setTimeout(() => node.remove(), ms);
  return node;
}

// ---- modal
export function modal(build) {
  const wrap = $("#modal"), card = $("#modalCard");
  card.innerHTML = "";
  const close = () => { wrap.classList.add("hidden"); card.innerHTML = ""; document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  wrap.onclick = (e) => { if (e.target === wrap) close(); };
  build(card, close);
  wrap.classList.remove("hidden");
  return close;
}

export function confirm(message, { ok = "OK", cancel = "Cancel", danger = false } = {}) {
  return new Promise((resolve) => {
    modal((card, close) => {
      card.append(
        el("div", { style: { fontSize: "14px", lineHeight: "1.5" }, text: message }),
        el("div", { class: "actions" }, [
          el("button", { class: "btn", onclick: () => { close(); resolve(false); } }, [cancel]),
          el("button", { class: `btn ${danger ? "danger" : "primary"}`, onclick: () => { close(); resolve(true); } }, [ok]),
        ]),
      );
    });
  });
}
