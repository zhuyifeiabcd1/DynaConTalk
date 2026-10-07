"""LLM edit assistant: an external language model proposes keypose insertions for a job.

It only suggests. Every proposal lands in the Studio's pending edits, to be reviewed, adjusted
and applied by the user; the web server makes one HTTP call and never touches the model or the GPU.

The LLM receives
  1. the word timeline (features/words.json from webui/transcribe.py): frame range of each word;
  2. what it needs from the audio as text: the loudness of each word (RMS of the 16 kHz wav)
     and the pauses between words (the waveform itself is not sent; text-only models refuse audio);
  3. the keypose library (id / category / name / tags).

It only answers "which keypose at which frame": part and strength are fixed (EDIT_DEFAULTS).
A small choice space keeps the proposals usable.

Any OpenAI-compatible (/v1/chat/completions) or Anthropic-compatible (/v1/messages) endpoint
works. The settings (base_url / model / api_key) are stored in <output_root>/agent_config.json
with mode 0600; the browser only sees the last four characters of the key.
"""
from __future__ import annotations

import json
import os
import re
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "EDIT_DEFAULTS", "DEFAULT_CONFIG", "load_config", "save_config", "public_config",
    "build_catalog", "word_loudness", "build_context", "build_prompt",
    "call_llm", "parse_edits", "AgentError",
]

# every proposed edit uses these settings: the assistant picks poses and moments, not parameters
EDIT_DEFAULTS = {"part": "full_body", "strength": 1.0, "sigma": 6.0}

DEFAULT_CONFIG: Dict[str, Any] = {
    "base_url": os.environ.get("STUDIO_AGENT_BASE_URL", ""),
    "model": os.environ.get("STUDIO_AGENT_MODEL", ""),
    "protocol": os.environ.get("STUDIO_AGENT_PROTOCOL", "auto"),   # auto | openai | anthropic
    "api_key": os.environ.get("STUDIO_AGENT_KEY", ""),
    "max_edits": 12,
    "min_gap_frames": 45,          # 1.5 s at 30 fps: closer edits would regenerate overlapping windows
    "temperature": None,           # None: not sent, the server default applies
    "timeout": 180,
    "max_output_tokens": 4096,
}

USER_AGENT = "DynaConTalk-Studio/1.0"


class AgentError(RuntimeError):
    """A failure reason that can be shown to the user as is."""


# ------------------------------------------------------------------ settings
def load_config(path: Path) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    try:
        saved = json.loads(path.read_text("utf-8"))
        if isinstance(saved, dict):
            cfg.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(path: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config(path)
    for key in ("base_url", "model", "protocol"):
        if key in patch and patch[key] is not None:
            cfg[key] = str(patch[key]).strip()
    # an empty field keeps the stored key; null clears it
    if "api_key" in patch:
        raw = patch["api_key"]
        if raw is None:
            cfg["api_key"] = ""
        elif str(raw).strip():
            cfg["api_key"] = str(raw).strip()
    for key, lo, hi in (("max_edits", 1, 40), ("min_gap_frames", 0, 600),
                        ("timeout", 10, 900), ("max_output_tokens", 256, 32768)):
        if key in patch and patch[key] is not None:
            try:
                cfg[key] = int(np.clip(int(patch[key]), lo, hi))
            except (TypeError, ValueError):
                pass
    if "temperature" in patch:
        raw = patch["temperature"]
        if raw is None or str(raw).strip() == "":
            cfg["temperature"] = None          # empty: do not send the field
        else:
            try:
                cfg["temperature"] = float(np.clip(float(raw), 0.0, 2.0))
            except (TypeError, ValueError):
                pass
    if cfg["protocol"] not in ("auto", "openai", "anthropic"):
        cfg["protocol"] = "auto"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    os.chmod(tmp, 0o600)          # holds the API key
    tmp.replace(path)
    return cfg


def public_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """What can be sent to the browser: the key reduced to its last four characters."""
    key = str(cfg.get("api_key") or "")
    out = {k: v for k, v in cfg.items() if k != "api_key"}
    out["has_key"] = bool(key)
    out["key_hint"] = f"…{key[-4:]}" if len(key) >= 4 else ("…" if key else "")
    out["resolved_protocol"] = resolve_protocol(cfg)
    out["endpoint"] = endpoint_url(cfg)
    return out


def resolve_protocol(cfg: Dict[str, Any]) -> str:
    proto = str(cfg.get("protocol") or "auto")
    if proto in ("openai", "anthropic"):
        return proto
    base = str(cfg.get("base_url") or "").rstrip("/")
    # OpenAI-compatible endpoints are given as .../v1, Anthropic-compatible ones as the root URL
    return "openai" if base.endswith("/v1") or "/openai" in base else "anthropic"


def endpoint_url(cfg: Dict[str, Any]) -> str:
    base = str(cfg.get("base_url") or "").rstrip("/")
    if not base:
        return ""
    if resolve_protocol(cfg) == "openai":
        return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    return base + ("/messages" if base.endswith("/v1") else "/v1/messages")


# ------------------------------------------------------------------ prompt material
def build_catalog(keyposes: List[Dict[str, Any]]) -> str:
    """The keypose library, one line per pose: id | category | name | tags (about 12 KB)."""
    lines = []
    for item in keyposes:
        tags = ",".join(str(t) for t in (item.get("tags") or [])[:6])
        name = str(item.get("name") or item.get("id") or "").strip()
        cat = str(item.get("category") or "other")
        lines.append(f"{item['id']} | {cat} | {name}" + (f" | {tags}" if tags else ""))
    return "\n".join(lines)


def word_loudness(audio_path: Optional[Path], words: List[Dict[str, Any]]) -> Dict[int, float]:
    """Loudness of each word in 0-1 (normalized by the 5th / 95th percentile of this clip).

    This is how the audio reaches the prompt: the LLM gets no waveform, but it gets which words
    are stressed, and stress is where co-speech gestures land. Empty if the wav cannot be read.
    """
    if not audio_path or not Path(audio_path).exists() or not words:
        return {}
    try:
        with wave.open(str(audio_path), "rb") as wav:
            sr = wav.getframerate()
            n = wav.getnframes()
            raw = wav.readframes(n)
            ch = wav.getnchannels()
            width = wav.getsampwidth()
        if width != 2:
            return {}
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
    except (OSError, wave.Error, ValueError):
        return {}
    rms: Dict[int, float] = {}
    for w in words:
        s = int(float(w.get("start", 0.0)) * sr)
        e = int(float(w.get("end", 0.0)) * sr)
        seg = pcm[max(0, s):max(s + 1, min(len(pcm), e))]
        rms[int(w.get("i", len(rms)))] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    if not rms:
        return {}
    vals = np.array(list(rms.values()), dtype=np.float64)
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    span = max(hi - lo, 1e-6)
    return {k: float(np.clip((v - lo) / span, 0.0, 1.0)) for k, v in rms.items()}


def build_context(
    words_json: Optional[Dict[str, Any]],
    transcripts: Optional[Dict[str, Any]],
    total_frames: int,
    fps: int,
    audio_path: Optional[Path] = None,
    max_words: int = 900,
) -> Tuple[str, Dict[str, Any]]:
    """The speech part of the prompt (transcript, word timings, loudness, pauses) and its statistics."""
    meta: Dict[str, Any] = {"words": 0, "sentences": 0, "pauses": 0, "loudness": False, "timed": False}
    duration = total_frames / float(fps or 30)
    head = [f"Clip: {total_frames} frames at {fps} fps ({duration:.1f} s). Valid frame range: 0–{max(0, total_frames - 1)}."]

    words = list((words_json or {}).get("words") or [])
    sentences = list((words_json or {}).get("sentences") or [])
    if not words:
        # no word alignment: only the per-chunk text; still usable, far less precise
        rows = (transcripts or {}).get("chunks") or (transcripts if isinstance(transcripts, list) else [])
        text = " ".join(str(r.get("text") or r) if isinstance(r, dict) else str(r) for r in (rows or []))
        head.append("\nTranscript (NO per-word timing available — place edits by your own estimate of pacing):")
        head.append(text.strip() or "(no transcript)")
        return "\n".join(head), meta

    meta.update({"words": len(words), "sentences": len(sentences), "timed": True})
    loud = word_loudness(audio_path, words)
    meta["loudness"] = bool(loud)

    head.append("\nSentences (frame ranges):")
    for s in sentences:
        head.append(f"  [{int(s.get('sf', 0))}–{int(s.get('ef', 0)) - 1}] {str(s.get('text') or '').strip()}")

    step = max(1, len(words) // max_words + 1)
    head.append(
        "\nWords: frame_start-frame_end  word" + ("  (loud=0..1 relative emphasis)" if loud else "")
        + (f"   [every {step}th word shown]" if step > 1 else "")
    )
    for w in words[::step]:
        mark = f"  loud={loud.get(int(w.get('i', -1)), 0.0):.2f}" if loud else ""
        head.append(f"  {int(w.get('sf', 0))}-{int(w.get('ef', 0)) - 1}  {str(w.get('text') or '')}{mark}")

    # pauses between words: where gestures usually start or settle
    pauses = []
    for a, b in zip(words, words[1:]):
        gap = int(b.get("sf", 0)) - int(a.get("ef", 0))
        if gap >= max(6, int(0.25 * fps)):
            pauses.append((int(a.get("ef", 0)), int(b.get("sf", 0)), str(a.get("text") or "")))
    meta["pauses"] = len(pauses)
    if pauses:
        head.append("\nPauses (silence between words — natural places for a gesture to land or settle):")
        for sf, ef, after in pauses[:60]:
            head.append(f"  {sf}-{ef}  ({(ef - sf) / fps:.2f} s, after \"{after}\")")
    return "\n".join(head), meta


SYSTEM_PROMPT = """You are a co-speech gesture director for a 3D speaker animation tool.

You place body keyposes on a timeline. Each edit you output makes the character pass \
through one pose from a fixed library at one frame; the animation system blends it into \
the surrounding motion automatically.

Rules:
- Use ONLY keypose ids from the catalogue given to you. Never invent an id.
- One edit = one keypose id + one frame number. Nothing else is yours to choose.
- Place gestures where speech justifies them: on stressed words, on the beat that opens a \
new idea, on enumerations, on contrasts, on the phrase before a pause. Not on silence, and \
not on unstressed function words.
- Gestures need room. Keep consecutive edits at least the requested minimum apart; a real \
speaker holds a pose for roughly one phrase.
- Prefer variety. Repeating one pose across a clip looks mechanical.
- Match meaning to pose: pointing for deixis and naming, open palms for offering or \
explaining, counting for enumeration, raised or wide for emphasis and scale, chest or clasp \
for personal or earnest statements, rest for closing a thought.
- Fewer, well-placed gestures beat many. If the speech does not justify the maximum, output less.

Reply with JSON only, no prose, no code fence:
{"edits": [{"keypose_id": "keypose_012", "frame": 143, "reason": "beat on 'never'"}], "summary": "one sentence"}
The reason must be short and name the word or phrase it lands on."""


def build_prompt(
    catalog: str,
    context: str,
    instruction: str,
    max_edits: int,
    min_gap_frames: int,
    fps: int,
) -> Tuple[str, str]:
    ask = [
        "# Keypose catalogue (id | category | name | tags)",
        catalog,
        "",
        "# Speech",
        context,
        "",
        "# Task",
        f"Propose at most {max_edits} keypose insertions for this clip.",
        f"Consecutive edits must be at least {min_gap_frames} frames apart ({min_gap_frames / max(1, fps):.1f} s).",
    ]
    if instruction.strip():
        ask += ["", "# Director's note (follow it)", instruction.strip()]
    ask += ["", "Reply with the JSON object only."]
    return SYSTEM_PROMPT, "\n".join(ask)


# ------------------------------------------------------------------ HTTP call
def call_llm(cfg: Dict[str, Any], system: str, user: str) -> Dict[str, Any]:
    """Blocking call (run in a thread). Returns {text, usage, model, endpoint, protocol}."""
    import httpx

    key = str(cfg.get("api_key") or "")
    url = endpoint_url(cfg)
    if not key:
        raise AgentError("No API key yet: set it in the AI assistant settings")
    if not url:
        raise AgentError("No API base URL yet: set it in the AI assistant settings")
    proto = resolve_protocol(cfg)
    timeout = float(cfg.get("timeout") or 180)
    max_tok = int(cfg.get("max_output_tokens") or 4096)
    temp = cfg.get("temperature")

    if proto == "openai":
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": USER_AGENT}
        payload = {
            "model": cfg.get("model"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tok,
        }
    else:
        headers = {
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "Content-Type": "application/json", "User-Agent": USER_AGENT,
        }
        payload = {
            "model": cfg.get("model"),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tok,
        }
    if temp is not None:
        payload["temperature"] = float(temp)

    def _post(body):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                return client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise AgentError(f"Request timed out ({timeout:.0f} s): {url}") from exc
        except httpx.HTTPError as exc:
            raise AgentError(f"Cannot reach {url}: {exc}") from exc

    res = _post(payload)
    # some servers accept only their own temperature: retry without it
    if res.status_code == 400 and "temperature" in res.text.lower() and "temperature" in payload:
        payload.pop("temperature")
        res = _post(payload)

    if res.status_code >= 400:
        detail = res.text[:400].replace("\n", " ")
        hint = ""
        if res.status_code in (401, 403):
            hint = " (wrong key, or no access to this model)"
        elif res.status_code == 404:
            hint = " (wrong URL or model name: OpenAI protocol uses .../v1, Anthropic protocol the root URL)"
        elif res.status_code == 429:
            hint = " (rate limited, try again later)"
        raise AgentError(f"HTTP {res.status_code}{hint}: {detail}")

    try:
        data = res.json()
    except ValueError as exc:
        raise AgentError(f"The reply is not JSON: {res.text[:200]}") from exc

    if proto == "openai":
        choices = data.get("choices") or []
        if not choices:
            raise AgentError(f"No choices in the reply: {json.dumps(data)[:300]}")
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):     # some servers return the content in parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    else:
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")

    if not str(text).strip():
        raise AgentError(f"Empty reply (stop_reason={data.get('stop_reason') or data.get('finish_reason')})")
    return {
        "text": str(text),
        "usage": data.get("usage") or {},
        "model": data.get("model") or cfg.get("model"),
        "endpoint": url,
        "protocol": proto,
    }


# ------------------------------------------------------------------ parsing and validation
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> Any:
    raw = text.strip()
    fence = _FENCE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except ValueError:
            pass
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except ValueError:
            pass
    raise AgentError(f"No parsable JSON in the reply: {text[:300]}")


def parse_edits(
    text: str,
    valid_ids: set,
    total_frames: int,
    fps: int,
    max_edits: int,
    min_gap_frames: int,
) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """Turn the model's reply into valid edits. Returns (edits, reasons for dropped items, summary).

    The reply is not trusted: ids must be in the library, frames in range, an edit closer than
    min_gap_frames to the previous one is dropped, and the list is cut at max_edits.
    """
    data = _extract_json(text)
    if isinstance(data, list):
        items, summary = data, ""
    elif isinstance(data, dict):
        items = data.get("edits") or data.get("keyposes") or data.get("items") or []
        summary = str(data.get("summary") or data.get("note") or "")[:400]
    else:
        raise AgentError("Unrecognized JSON structure in the reply")
    if not isinstance(items, list):
        raise AgentError("'edits' is not a list")

    notes: List[str] = []
    staged: List[Dict[str, Any]] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            notes.append(f"#{i + 1} is not an object, skipped")
            continue
        kid = str(raw.get("keypose_id") or raw.get("id") or raw.get("keypose") or "").strip()
        if kid not in valid_ids:
            notes.append(f"#{i + 1} unknown keypose '{kid or '-'}', skipped")
            continue
        frame = raw.get("frame")
        if frame is None and raw.get("time_sec") is not None:
            try:
                frame = float(raw["time_sec"]) * fps
            except (TypeError, ValueError):
                frame = None
        try:
            frame = int(round(float(frame)))
        except (TypeError, ValueError):
            notes.append(f"#{i + 1} invalid frame, skipped")
            continue
        if total_frames and not (0 <= frame < total_frames):
            notes.append(f"#{i + 1} frame {frame} outside 0-{total_frames - 1}, skipped")
            continue
        staged.append({
            "keypose_id": kid,
            "frame": max(0, frame),
            "reason": str(raw.get("reason") or raw.get("why") or "")[:160],
        })

    staged.sort(key=lambda e: e["frame"])
    kept: List[Dict[str, Any]] = []
    for item in staged:
        if kept and item["frame"] - kept[-1]["frame"] < min_gap_frames:
            notes.append(f"f{item['frame']} is less than {min_gap_frames} frames after the previous edit, dropped")
            continue
        if len(kept) >= max_edits:
            notes.append(f"more than {max_edits} edits, list cut")
            break
        kept.append({**item, **EDIT_DEFAULTS})
    return kept, notes, summary
