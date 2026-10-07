"""DynaConTalk Studio - web front-end for DynaConTalk generation and editing.

* The web layer never imports model code.  Every job is a directory under
  ``outputs/studio/jobs/<job_id>`` holding ``request.json`` / ``status.json`` /
  ``events.jsonl`` / ``artifacts.json`` plus the outputs of the job process
  (``python -m webui.generate | webui.edit | webui.transcribe``).
* A single in-process worker thread runs one job process at a time (single GPU).
  The worker waits for free GPU memory before launching.
* The browser talks JSON; the page itself is a static single-page app served
  from ``webui/static``.

Run from the repository root::

    uvicorn webui.app:app --host 127.0.0.1 --port 7860
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui import traj_script as ts  # noqa: E402  (numpy only)
from webui import agent as agent_mod  # noqa: E402  (httpx only, no model code)

APP_VERSION = "1.0.0"
FPS = 30
OUTPUT_ROOT = Path(os.environ.get("STUDIO_OUTPUT_ROOT") or ROOT / "outputs" / "studio")
JOBS_DIR = OUTPUT_ROOT / "jobs"
ASSETS_DIR = Path(os.environ.get("STUDIO_ASSET_ROOT") or ROOT / "assets")

JOB_MODULES = {"generate": "webui.generate", "edit": "webui.edit", "transcribe": "webui.transcribe"}
PYTHON_CMD = shlex.split(os.environ.get("STUDIO_PYTHON", "")) or [sys.executable]
MIN_FREE_GPU_MB = int(os.environ.get("STUDIO_MIN_FREE_MB", "6000"))  # a generation peaks at about 5.3 GB
RENDER_AVAILABLE = os.environ.get("STUDIO_RENDER", "1") != "0"  # set to 0 by the launcher without off-screen OpenGL
GPU_INDEX = os.environ.get("STUDIO_GPU", "0")
# Credentials of the LLM edit assistant (base_url / model / api_key), file mode 0600; never served.
AGENT_CONFIG_PATH = Path(os.environ.get("STUDIO_AGENT_CONFIG") or OUTPUT_ROOT / "agent_config.json")

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
ACTIVE_STATES = {"queued", "running", "cancelling", "waiting"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MOTION_EXT = {".npy", ".npz"}
ASSET_KINDS = ("identities", "trajectories", "keyposes")

KEYPOSE_CATEGORIES = [
    "raise", "wide", "open_palms", "side", "chest", "point", "reach",
    "fist", "clasp", "fold", "head", "behind", "lean", "count", "rest", "other",
]
# trajectory taxonomy: how big the movement is, and how often it happens
TRAJECTORY_CATEGORIES = ["anchored", "subtle", "moderate", "active", "wide"]
TRAJECTORY_FREQUENCIES = ["steady", "rhythmic", "restless"]
BODY_PARTS = ["right_arm", "left_arm", "both_arms", "hands", "torso", "upper_body", "full_body"]
WAVELET_BANDS = ["ca3", "cd3", "cd2", "cd1"]


# =============================================================================== helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def safe_filename(name: str) -> str:
    base = Path(name or "upload.bin").name
    cleaned = re.sub(r"[^A-Za-z0-9_.\-一-鿿]+", "_", base).strip("._")
    return cleaned or "upload.bin"


def make_job_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def file_url(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(ROOT.resolve()).as_posix()
        return "/api/files/" + quote(rel, safe="/")
    except ValueError:
        return "/api/files/" + quote(str(resolved), safe="/")


def allowed_file_roots() -> List[Path]:
    """Files served by /api/files: job outputs and the asset libraries only."""
    return [JOBS_DIR.resolve(), ASSETS_DIR.resolve()]


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def job_dir(job_id: str) -> Path:
    if not SAFE_ID.match(job_id or ""):
        raise HTTPException(status_code=400, detail="Invalid job id")
    return JOBS_DIR / job_id


def tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


# =============================================================================== GPU / system
def gpu_stats() -> List[Dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mb": int(float(parts[2])),
            "memory_total_mb": int(float(parts[3])),
            "utilization": int(float(parts[4])),
        })
    return gpus


def gpu_free_mb(index: str = GPU_INDEX) -> Optional[int]:
    for g in gpu_stats():
        if str(g["index"]) == str(index):
            return g["memory_total_mb"] - g["memory_used_mb"]
    return None


# =============================================================================== assets
_ASSET_CACHE: Dict[str, Any] = {}


def asset_root(kind: str) -> Path:
    return ASSETS_DIR / kind


def overlay_path(kind: str) -> Path:
    return asset_root(kind) / "metadata.json"


def _resolve_media(record: Dict[str, Any], root: Path) -> Optional[Path]:
    """Cover image (keyposes, identities) or path thumbnail (trajectories), relative to the library."""
    for key in ("cover", "thumb"):
        value = record.get(key)
        if value and (root / str(value)).exists():
            return (root / str(value)).resolve()
    return None


def _auto_keypose_label(record: Dict[str, Any]) -> str:
    speaker = record.get("speaker") or _speaker_from_key(record.get("sample_key"))
    t = record.get("time_sec")
    if speaker and t is not None:
        return f"{speaker} · {float(t):.1f}s"
    return str(record.get("id") or "keypose")


def _speaker_from_key(sample_key: Any) -> str:
    parts = str(sample_key or "").split("_")
    return parts[1] if len(parts) >= 2 else ""


def _speaker_key(sample_key: Any) -> str:
    """`12_zhao_0_1_1` -> `12_zhao`; pairs trajectories with identities (names may repeat, ids do not)."""
    parts = str(sample_key or "").split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else ""


def _rule_tags(record: Dict[str, Any]) -> List[str]:
    """Coarse tags derived from the mining scores (used when no annotation exists)."""
    scores = record.get("scores") or {}
    tags: List[str] = []
    hh = scores.get("hand_height")
    hs = scores.get("hand_spread")
    asym = scores.get("asymmetry")
    if hh is not None:
        tags.append("hands_high" if hh > 0.05 else ("hands_low" if hh < -0.2 else "hands_mid"))
    if hs is not None:
        tags.append("open" if hs > 0.8 else ("narrow" if hs < 0.55 else "medium"))
    if asym is not None:
        tags.append("one_hand" if asym > 0.4 else "two_hands")
    return tags


def list_assets(kind: str) -> List[Dict[str, Any]]:
    if kind not in ASSET_KINDS:
        raise HTTPException(status_code=404, detail="Unknown asset kind")
    root = asset_root(kind)
    if not root.exists():
        return []
    stamp = (
        (root / "manifest.jsonl").stat().st_mtime if (root / "manifest.jsonl").exists() else 0,
        overlay_path(kind).stat().st_mtime if overlay_path(kind).exists() else 0,
    )
    cached = _ASSET_CACHE.get(kind)
    if cached and cached[0] == stamp:
        return cached[1]

    overlay = read_json(overlay_path(kind), {})
    items: List[Dict[str, Any]] = []
    seen = set()
    for record in read_jsonl(root / "manifest.jsonl"):
        item_id = str(record.get("id") or record.get("name") or record.get("sample_key") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        meta = overlay.get(item_id, {}) if isinstance(overlay, dict) else {}
        media = _resolve_media(record, root)
        speaker = record.get("speaker") or record.get("speaker_name") or _speaker_from_key(record.get("sample_key"))
        item: Dict[str, Any] = {
            "id": item_id,
            "kind": kind,
            "name": meta.get("name") or record.get("name") or (_auto_keypose_label(record) if kind == "keyposes" else item_id),
            "name_zh": meta.get("name_zh") or record.get("name_zh") or "",
            "category": meta.get("category") or record.get("category") or ("other" if kind == "keyposes" else ""),
            "tags": list(meta.get("tags") or record.get("tags") or []),
            "notes": meta.get("notes") or "",
            "speaker": speaker,
            "sample_key": record.get("sample_key") or record.get("sample_id") or "",
            # how much a person moves is mostly a property of the speaker, so trajectories belong
            # to an identity; the front end pairs them through this key
            "speaker_key": _speaker_key(record.get("sample_key") or record.get("sample_id") or ""),
            "media_url": file_url(media) if media else "",
            "path": str(record.get("path") or record.get("pose") or ""),
        }
        if kind == "keyposes":
            item.update({
                "frame": record.get("frame"),
                "time_sec": record.get("time_sec"),
                "fps": record.get("fps", FPS),
                "scores": record.get("scores") or {},
                "auto_tags": _rule_tags(record),
                "part_tags": record.get("part_tags") or [],
            })
        elif kind == "trajectories":
            motion = meta.get("motion") or {}
            item.update({
                "frames": record.get("frames"),
                "duration_sec": record.get("duration_sec"),
                "features": record.get("features") or {},
                "quality_score": record.get("quality_score"),
                "motion": motion,
                "frequency": motion.get("frequency") or "",
                "amplitude_score": motion.get("amplitude_score"),
                "is_default": bool(meta.get("default")),
            })
            if not item["category"]:
                item["category"] = motion.get("amplitude") or "moderate"
        elif kind == "identities":
            item.update({
                "speaker_id": record.get("speaker_id"),
                "representative_sample": record.get("representative_sample") or record.get("sample_key"),
                # measured on the SMPL-X mesh of the identity's body shape
                "stature_m": record.get("stature_m"),
                "shoulder_width_m": record.get("shoulder_width_m"),
            })
        items.append(item)
    _ASSET_CACHE[kind] = (stamp, items)
    return items


def trajectories_for_identity(identity_id: str) -> List[Dict[str, Any]]:
    """Trajectories recorded from the identity's speaker (all of them for an empty id)."""
    items = list_assets("trajectories")
    if not identity_id:
        return items
    ident = next((i for i in list_assets("identities") if i["id"] == identity_id), None)
    if not ident:
        return items
    key = ident.get("speaker_key") or ""
    return [t for t in items if t.get("speaker_key") == key] or items


def default_trajectory_id(identity_id: str = "") -> str:
    """The calmest trajectory of the identity's own speaker; without an identity, the library default.

    How much a person moves is mostly a property of the speaker, so a calm trajectory of another
    speaker would ask the model for motion it has not seen for this identity.
    """
    items = trajectories_for_identity(identity_id)
    if identity_id:
        pool = sorted(items, key=lambda i: (i.get("amplitude_score") if i.get("amplitude_score") is not None else 0.0))
        return pool[0]["id"] if pool else ""
    for item in items:
        if item.get("is_default"):
            return item["id"]
    calm = [i for i in items if i.get("category") == "anchored"]
    pool = calm or items
    pool = sorted(pool, key=lambda i: (i.get("amplitude_score") if i.get("amplitude_score") is not None else 0.0))
    return pool[0]["id"] if pool else ""


def get_asset(kind: str, asset_id: str) -> Dict[str, Any]:
    for item in list_assets(kind):
        if item["id"] == asset_id:
            return item
    raise HTTPException(status_code=404, detail=f"{kind[:-1]} '{asset_id}' not found")


def update_asset_overlay(kind: str, asset_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    get_asset(kind, asset_id)
    allowed = {"name", "name_zh", "category", "tags", "notes"}
    clean: Dict[str, Any] = {}
    for key, value in patch.items():
        if key not in allowed:
            continue
        if key == "tags":
            if isinstance(value, str):
                value = [t.strip() for t in re.split(r"[,;，；\s]+", value) if t.strip()]
            clean[key] = [str(t)[:40] for t in (value or [])][:20]
        elif key == "category":
            clean[key] = str(value or "other")[:40]
        else:
            clean[key] = str(value or "")[:200]
    path = overlay_path(kind)
    overlay = read_json(path, {})
    if not isinstance(overlay, dict):
        overlay = {}
    entry = dict(overlay.get(asset_id, {}))  # keeps computed blocks (motion, default)
    entry.update(clean)
    entry["updated_at"] = utc_now()
    overlay[asset_id] = entry
    atomic_write_json(path, overlay)
    _ASSET_CACHE.pop(kind, None)
    return get_asset(kind, asset_id)


def trajectory_asset_arrays(asset_id: str) -> Dict[str, np.ndarray]:
    item = get_asset("trajectories", asset_id)
    path = asset_root("trajectories") / item["path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"trajectory file missing: {path}")
    z = np.load(path)
    out = {k: np.asarray(z[k]) for k in z.files}
    return out


def trajectory_base(asset_id: str) -> Dict[str, Any]:
    """Base delta sequence + integration anchors for a trajectory asset."""
    arrays = trajectory_asset_arrays(asset_id)
    if "trajectory" in arrays and arrays["trajectory"].ndim == 2 and arrays["trajectory"].shape[1] == 4:
        delta = arrays["trajectory"].astype(np.float32)
        init_pos = arrays.get("init_pos", np.zeros((1, 3), np.float32)).reshape(3)
        init_yaw = float(np.asarray(arrays.get("init_yaw", [0.0])).reshape(-1)[0])
    else:
        trans = arrays["trans"].astype(np.float32)
        delta, init_pos, init_yaw = ts.deltas_from_trans_root(trans, np.zeros((trans.shape[0], 3), np.float32))
    return {"delta": delta, "init_pos": np.asarray(init_pos, np.float32), "init_yaw": init_yaw}


# =============================================================================== job files
_NPY_CACHE: Dict[str, Any] = {}


def load_chunk_obj(path: Path) -> Dict[str, Any]:
    key = str(path)
    mtime = path.stat().st_mtime
    cached = _NPY_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    obj = np.load(path, allow_pickle=True).item()
    if len(_NPY_CACHE) > 24:
        _NPY_CACHE.clear()
    _NPY_CACHE[key] = (mtime, obj)
    return obj


def job_base_trajectory(job_id: str, chunk_index: int) -> Dict[str, Any]:
    """Base delta sequence of the trajectory condition used for a job chunk."""
    detail = job_detail(job_id)
    chunk = next((c for c in detail["chunks"] if c["index"] == chunk_index), None)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    npy = chunk.get("npy_path")
    if not npy or not Path(npy).exists():
        raise HTTPException(status_code=404, detail="chunk motion file missing")
    obj = load_chunk_obj(Path(npy))
    motion = np.asarray(obj["motion"], dtype=np.float32)
    trans = np.asarray(obj.get("condition_trans", motion[:, 165:168]), dtype=np.float32)
    root = np.asarray(obj.get("condition_root_orient", motion[:, :3]), dtype=np.float32)
    n = min(trans.shape[0], root.shape[0])
    delta, init_pos, init_yaw = ts.deltas_from_trans_root(trans[:n], root[:n])
    return {"delta": delta, "init_pos": init_pos, "init_yaw": init_yaw}


# =============================================================================== job manager
class JobManager:
    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen[Any]] = {}
        self._started = False
        self._worker: Optional[threading.Thread] = None
        self.current_job: Optional[str] = None
        self._list_version = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            self._recover()
            self._worker = threading.Thread(target=self._loop, name="studio-job-worker", daemon=True)
            self._worker.start()
            self._started = True

    def _recover(self) -> None:
        for status in self.list_statuses():
            state = status.get("state")
            jid = str(status.get("job_id") or "")
            if not jid:
                continue
            if state in {"queued", "waiting"}:
                self._queue.put(jid)
            elif state in {"running", "cancelling"}:
                self._set_status(jid, state="failed", phase="error", message="Server restarted while the job was active.",
                                 finished_at=utc_now(), progress=1.0)

    def bump(self) -> None:
        self._list_version += 1

    @property
    def list_version(self) -> int:
        return self._list_version

    # ------------------------------------------------------------------ CRUD
    def create(self, action: str, payload: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
        self.start()
        jid = job_id or make_job_id()
        directory = job_dir(jid)
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        request = {"job_id": jid, "action": action, "created_at": utc_now(), **payload}
        status = {
            "job_id": jid, "action": action, "state": "queued", "phase": "queued", "progress": 0.0,
            "message": "Queued", "created_at": request["created_at"], "updated_at": utc_now(),
            "started_at": None, "finished_at": None, "pid": None, "returncode": None,
            "cancel_requested": False, "title": payload.get("title") or "", "parent_id": payload.get("source_job_id") or "",
        }
        atomic_write_json(directory / "request.json", request)
        atomic_write_json(directory / "status.json", status)
        atomic_write_json(directory / "artifacts.json", {"job_id": jid, "items": [], "updated_at": utc_now()})
        self._event(jid, "queued", {"status": status})
        self._queue.put(jid)
        self.bump()
        return status

    def cancel(self, jid: str) -> Dict[str, Any]:
        self.start()
        status = self.status(jid)
        if status.get("state") in TERMINAL_STATES:
            return status
        with self._lock:
            proc = self._processes.get(jid)
            if proc and proc.poll() is None:
                self._terminate(proc)
                self._set_status(jid, state="cancelling", cancel_requested=True, message="Cancelling (SIGTERM sent)")
            else:
                self._set_status(jid, state="cancelled", phase="cancelled", cancel_requested=True,
                                 message="Cancelled before start", finished_at=utc_now(), progress=1.0)
        self.bump()
        return self.status(jid)

    def purge(self, jid: str) -> None:
        status = self.status(jid)
        if status.get("state") not in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="Cancel the job before deleting it")
        shutil.rmtree(job_dir(jid), ignore_errors=True)
        self.bump()

    def patch(self, jid: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"title", "notes", "starred"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "title" in updates:
            updates["title"] = str(updates["title"] or "")[:120]
        if "starred" in updates:
            updates["starred"] = bool(updates["starred"])
        if updates:
            self._set_status(jid, **updates)
            req_path = job_dir(jid) / "request.json"
            req = read_json(req_path, {})
            req.update({k: v for k, v in updates.items() if k in {"title", "notes"}})
            atomic_write_json(req_path, req)
            self.bump()
        return self.status(jid)

    def status(self, jid: str) -> Dict[str, Any]:
        path = job_dir(jid) / "status.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        status = read_json(path)
        status.setdefault("job_id", jid)
        return status

    def request(self, jid: str) -> Dict[str, Any]:
        return read_json(job_dir(jid) / "request.json", {})

    def list_statuses(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        statuses = []
        for path in JOBS_DIR.glob("*/status.json"):
            st = read_json(path)
            if st:
                st.setdefault("job_id", path.parent.name)
                statuses.append(st)
        statuses.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
        return statuses[:limit] if limit else statuses

    def logs(self, jid: str, offset: int = 0, max_bytes: int = 400_000) -> Dict[str, Any]:
        directory = job_dir(jid)
        if not directory.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        path = directory / "logs" / "process.log"
        if not path.exists():
            return {"text": "", "offset": 0, "size": 0}
        size = path.stat().st_size
        offset = max(0, min(int(offset), size))
        start = max(offset, size - max_bytes) if offset == 0 else offset
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read()
        return {"text": data.decode("utf-8", errors="replace"), "offset": size, "size": size, "truncated": start > offset}

    def events_path(self, jid: str) -> Path:
        directory = job_dir(jid)
        if not directory.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        return directory / "events.jsonl"

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            jid = self._queue.get()
            try:
                if self.status(jid).get("state") == "cancelled":
                    continue
                if str(read_json(job_dir(jid) / "request.json", {}).get("action", "")) != "transcribe":
                    self._wait_for_gpu(jid)
                if self.status(jid).get("state") == "cancelled":
                    continue
                self._run(jid)
            except HTTPException:
                pass
            except Exception as exc:  # noqa: BLE001
                try:
                    self._set_status(jid, state="failed", phase="error", message=f"Worker error: {exc}",
                                     finished_at=utc_now(), progress=1.0)
                except Exception:
                    pass
            finally:
                self._queue.task_done()
                self.bump()

    def _wait_for_gpu(self, jid: str) -> None:
        if MIN_FREE_GPU_MB <= 0:
            return
        while True:
            free = gpu_free_mb()
            if free is None or free >= MIN_FREE_GPU_MB:
                return
            if self.status(jid).get("cancel_requested"):
                return
            msg = f"Waiting for GPU: {free / 1024:.1f} GB free, need {MIN_FREE_GPU_MB / 1024:.0f} GB"
            self._set_status(jid, state="waiting", phase="gpu", message=msg)
            self.bump()
            time.sleep(20)

    @staticmethod
    def _command(jid: str, action: str) -> List[str]:
        directory = job_dir(jid)
        return [*PYTHON_CMD, "-m", JOB_MODULES[action], "--job-dir", str(directory),
                "--request-json", str(directory / "request.json"), "--status-json", str(directory / "status.json"),
                "--artifacts-json", str(directory / "artifacts.json")]

    def _run(self, jid: str) -> None:
        directory = job_dir(jid)
        request = read_json(directory / "request.json")
        command = self._command(jid, str(request.get("action", "generate")))
        log_path = directory / "logs" / "process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "STUDIO_ASSET_ROOT": str(ASSETS_DIR)})
        if GPU_INDEX != "":
            env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
        self.current_job = jid
        self._set_status(jid, state="running", phase="starting", message="Starting", started_at=utc_now(), progress=0.02)
        self._event(jid, "process_start", {"command": command})
        self.bump()
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{utc_now()}] $ {' '.join(shlex.quote(a) for a in command)}\n")
            log_file.flush()
            kwargs: Dict[str, Any] = {"preexec_fn": os.setsid} if hasattr(os, "setsid") else {}
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True, **kwargs)
            with self._lock:
                self._processes[jid] = proc
            self._set_status(jid, pid=proc.pid)
            rc = proc.wait()
        with self._lock:
            self._processes.pop(jid, None)
        self.current_job = None
        latest = self.status(jid)
        cancelled = bool(latest.get("cancel_requested")) or latest.get("state") == "cancelling"
        collect_artifacts(jid)
        if cancelled:
            self._set_status(jid, state="cancelled", phase="cancelled", message="Cancelled", returncode=rc, finished_at=utc_now(), progress=1.0)
            self._event(jid, "cancelled", {"returncode": rc})
        elif rc == 0:
            self._set_status(jid, state="succeeded", phase="done", message="Completed", returncode=rc, finished_at=utc_now(), progress=1.0)
            self._event(jid, "succeeded", {"returncode": rc})
        else:
            msg = latest.get("message") if latest.get("state") == "failed" else f"Process exited with code {rc}"
            self._set_status(jid, state="failed", phase="error", message=msg, returncode=rc, finished_at=utc_now(), progress=1.0)
            self._event(jid, "failed", {"returncode": rc})

    # ------------------------------------------------------------------ status/events
    def _set_status(self, jid: str, **updates: Any) -> None:
        path = job_dir(jid) / "status.json"
        status = read_json(path)
        status.setdefault("job_id", jid)
        status.update(updates)
        status["updated_at"] = utc_now()
        atomic_write_json(path, status)
        self._event(jid, "status", {"status": status})

    def _event(self, jid: str, event: str, data: Dict[str, Any]) -> None:
        path = job_dir(jid) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": utc_now(), "event": event, "data": data}, ensure_ascii=False, default=_json_default) + "\n")

    def _terminate(self, proc: subprocess.Popen[Any]) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass


JOBS = JobManager()


# =============================================================================== job views
def _media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in MOTION_EXT:
        return "motion"
    if ext == ".json":
        return "json"
    return "file"


def _is_temp_render(path: Path) -> bool:
    return "features" in path.parts


def collect_artifacts(jid: str) -> List[Dict[str, Any]]:
    directory = job_dir(jid)
    if not directory.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or _is_temp_render(path.relative_to(directory)):
            continue
        if path.name in {"request.json", "status.json", "events.jsonl", "artifacts.json"} or path.parent.name == "logs":
            continue
        kind = _media_kind(path)
        if kind == "file":
            continue
        mime, _ = mimetypes.guess_type(path.name)
        items.append({
            "name": path.name, "rel": path.relative_to(directory).as_posix(), "path": str(path), "url": file_url(path),
            "size": path.stat().st_size, "mime": mime or "application/octet-stream", "kind": kind,
        })
    return items


def _chunk_video(directory: Path, index: int) -> Optional[Path]:
    """The rendered preview video of a chunk in a job directory."""
    path = directory / "renders" / "chunks" / f"chunk{index:03d}_body.mp4"
    return path if path.exists() else None


def _edited_chunks(directory: Path) -> set:
    """Chunks an edit job changed."""
    manifest = read_json(directory / "manifest.json", {}) if (directory / "request.json").exists() else {}
    return {int(seg.get("chunk_index", -1)) for seg in manifest.get("edit_segments") or []}


def job_summary(status: Dict[str, Any], request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    jid = status.get("job_id", "")
    request = request if request is not None else read_json(JOBS_DIR / jid / "request.json", {})
    started, finished = parse_iso(status.get("started_at")), parse_iso(status.get("finished_at"))
    elapsed = None
    if started:
        elapsed = ((finished or datetime.now(timezone.utc)) - started).total_seconds()
    audio = request.get("audio") or {}
    return {
        "job_id": jid,
        "action": status.get("action") or request.get("action") or "generate",
        "state": status.get("state", "unknown"),
        "phase": status.get("phase", ""),
        "progress": float(status.get("progress") or 0.0),
        "message": status.get("message", ""),
        "created_at": status.get("created_at") or request.get("created_at"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "elapsed_seconds": elapsed,
        "title": status.get("title") or request.get("title") or "",
        "starred": bool(status.get("starred")),
        "parent_id": status.get("parent_id") or request.get("source_job_id") or "",
        "audio_name": audio.get("filename") if isinstance(audio, dict) else "",
        "identity": request.get("identity") or "",
        "trajectory": request.get("trajectory") or "",
        "keypose_count": len(request.get("keyposes") or []),
        "traj_count": len(request.get("traj_script") or []),
        "render": bool(request.get("render", True)),
        "notes": request.get("notes") or "",
    }


def _parent_chain(jid: str, limit: int = 20) -> List[str]:
    chain = []
    seen = set()
    cur = jid
    while cur and cur not in seen and len(chain) < limit:
        seen.add(cur)
        req = read_json(JOBS_DIR / cur / "request.json", {})
        parent = str(req.get("source_job_id") or "")
        if parent and (JOBS_DIR / parent).exists():
            chain.append(parent)
        cur = parent
    return chain


def _inherited_edits(chain: List[str]) -> Dict[str, Any]:
    """Edits of the ancestor jobs, nearest first, each tagged with its job and depth."""
    kp: List[Dict[str, Any]] = []
    tr: List[Dict[str, Any]] = []
    for depth, ancestor in enumerate(chain, start=1):
        req = read_json(JOBS_DIR / ancestor / "request.json", {})
        for item in req.get("keyposes") or []:
            kp.append({**item, "from_job": ancestor, "depth": depth})
        for item in req.get("traj_script") or []:
            if isinstance(item, dict):
                tr.append({**item, "from_job": ancestor, "depth": depth})
    return {"inherited_keyposes": kp, "inherited_traj": tr}


def job_detail(jid: str) -> Dict[str, Any]:
    status = JOBS.status(jid)
    request = JOBS.request(jid)
    directory = job_dir(jid)
    manifest = read_json(directory / "manifest.json", {})
    summary = job_summary(status, request)
    chain = _parent_chain(jid)
    root_job = chain[-1] if chain else jid
    root_manifest = manifest if root_job == jid else read_json(JOBS_DIR / root_job / "manifest.json", {})

    # ---- chunk table (frame ranges come from the root generation job)
    chunk_rows = root_manifest.get("chunks") or []
    total_frames = int(root_manifest.get("audio_frames") or (chunk_rows[-1]["frame_end"] if chunk_rows else 0))

    # ---- which directories hold videos / motion for each chunk (edit jobs fall back to parents)
    lineage = [jid, *chain]
    edited_here = _edited_chunks(directory) if request.get("action") == "edit" else set()
    chunks = []
    for row in chunk_rows:
        idx = int(row["index"])
        video = None
        video_owner = None
        for owner in lineage:
            video = _chunk_video(JOBS_DIR / owner, idx)
            if video:
                video_owner = owner
                break
        source_video = None
        for owner in chain:
            source_video = _chunk_video(JOBS_DIR / owner, idx)
            if source_video:
                break
        npy_path = None
        for owner in lineage:
            for sub in ("bigru", "raw"):
                cand = JOBS_DIR / owner / sub / f"chunk{idx:03d}.npy"
                if cand.exists():
                    npy_path = cand
                    break
            if npy_path:
                break
        audio = row.get("audio_path")
        chunks.append({
            "index": idx,
            "frame_start": int(row.get("frame_start", 0)),
            "frame_end": int(row.get("frame_end", 0)),
            "frames": int(row.get("frame_end", 0)) - int(row.get("frame_start", 0)),
            "video_url": file_url(video) if video else "",
            "video_owner": video_owner or "",
            "edited": idx in edited_here,
            "source_video_url": file_url(source_video) if source_video else "",
            "audio_url": file_url(audio) if audio else "",
            "npy_path": str(npy_path) if npy_path else "",
            "npy_url": file_url(npy_path) if npy_path else "",
        })

    audio_url = ""
    a16 = list((JOBS_DIR / root_job / "audio").glob("*_16k.wav"))
    if a16:
        audio_url = file_url(a16[0])

    return {
        **summary,
        "status": status,
        "request": request,
        "manifest": {k: v for k, v in manifest.items() if k not in {"chunks"}},
        "fps": FPS,
        "total_frames": total_frames,
        "chunk_frames": int(root_manifest.get("chunk_frames") or 0),
        "chunks": chunks,
        "audio_url": audio_url,
        "keyposes": list(request.get("keyposes") or []),
        "traj_script": list(request.get("traj_script") or []),
        # every edit is applied on top of its parent's output, so the ancestors' edits are already in
        # this motion; they are listed so that the timeline shows the whole edit history
        **_inherited_edits(chain),
        "artifacts": collect_artifacts(jid),
        "lineage": chain,
        "root_job_id": root_job,
        "warnings": manifest.get("warnings") or [],
        "words_available": (JOBS_DIR / root_job / "features" / "words.json").exists(),
        "transcribe_pending": any(
            s.get("state") in ACTIVE_STATES and s.get("action") == "transcribe"
            and str(read_json(JOBS_DIR / s["job_id"] / "request.json", {}).get("target_job_id", "")) == root_job
            for s in JOBS.list_statuses()
        ),
        "transcripts": read_json(JOBS_DIR / root_job / "transcripts.json", {}) if (JOBS_DIR / root_job / "transcripts.json").exists() else None,
    }


# =============================================================================== request validation
def validate_keyposes(raw: Any, total_frames: Optional[int]) -> List[Dict[str, Any]]:
    items = parse_json_field(raw, [])
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="keyposes must be a list")
    valid_ids = {k["id"] for k in list_assets("keyposes")}
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kid = str(item.get("keypose_id") or item.get("id_ref") or "")
        if kid not in valid_ids:
            raise HTTPException(status_code=400, detail=f"unknown keypose '{kid}'")
        frame = parse_int(item.get("frame"), -1)
        if frame < 0 or (total_frames and frame >= total_frames):
            raise HTTPException(status_code=400, detail=f"keypose frame {frame} out of range")
        part = str(item.get("part") or "both_arms")
        if part not in BODY_PARTS:
            raise HTTPException(status_code=400, detail=f"unknown body part '{part}'")
        bands = [b for b in (item.get("bands") or WAVELET_BANDS) if b in WAVELET_BANDS] or WAVELET_BANDS
        out.append({
            "id": str(item.get("id") or f"kp_{i}"),
            "keypose_id": kid,
            "frame": frame,
            "part": part,
            "strength": float(np.clip(parse_float(item.get("strength"), 0.8), 0.0, 1.0)),
            "sigma": float(np.clip(parse_float(item.get("sigma"), 6.0), 0.5, 60.0)),
            "bands": bands,
            "label": str(item.get("label") or "")[:80],
        })
    out.sort(key=lambda k: k["frame"])
    return out


def validate_traj_script(raw: Any, total_frames: Optional[int]) -> List[Dict[str, Any]]:
    items = parse_json_field(raw, [])
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="traj_script must be a list")
    try:
        segs = ts.normalize_script(items, total_frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out = []
    for i, seg in enumerate(segs):
        d = seg.to_dict()
        d["id"] = d["id"] or f"tr_{i}"
        d["description"] = ts.describe_segment(seg)
        out.append(d)
    return out


# =============================================================================== app
@asynccontextmanager
async def lifespan(_: FastAPI):
    JOBS.start()
    yield


app = FastAPI(title="DynaConTalk Studio", version=APP_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def client_config() -> Dict[str, Any]:
    return {
        "version": APP_VERSION,
        "root": str(ROOT),
        "fps": FPS,
        "body_parts": BODY_PARTS,
        "bands": WAVELET_BANDS,
        "keypose_categories": KEYPOSE_CATEGORIES,
        "trajectory_categories": TRAJECTORY_CATEGORIES,
        "trajectory_frequencies": TRAJECTORY_FREQUENCIES,
        "default_trajectory": default_trajectory_id(),
        "primitives": {k: {"unit": v["unit"], "default": v["default"], "label": v["label"], "channel": v["channel"]} for k, v in ts.PRIMITIVES.items()},
        # Jinja's tojson sorts object keys, so the display order travels separately
        "primitive_order": list(ts.PRIMITIVES.keys()),
        "step_length": ts.STEP_LENGTH,
        "min_free_gpu_mb": MIN_FREE_GPU_MB,
        "agent": {"defaults": {k: v for k, v in agent_mod.DEFAULT_CONFIG.items() if k != "api_key"},
                  "edit_defaults": agent_mod.EDIT_DEFAULTS},
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"config": client_config()})


@app.get("/api/config")
async def api_config() -> JSONResponse:
    return JSONResponse(client_config())


@app.get("/api/system")
async def api_system() -> JSONResponse:
    gpus = gpu_stats()
    try:
        usage = shutil.disk_usage(OUTPUT_ROOT if OUTPUT_ROOT.exists() else ROOT)
        disk = {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    except OSError:
        disk = {}
    statuses = JOBS.list_statuses()
    return JSONResponse({
        "gpus": gpus,
        "gpu_index": GPU_INDEX,
        "min_free_gpu_mb": MIN_FREE_GPU_MB,
        "queue": [s["job_id"] for s in statuses if s.get("state") in {"queued", "waiting"}],
        "current_job": JOBS.current_job,
        "disk": disk,
        "time": utc_now(),
    })


# ---------------------------------------------------------------- assets
@app.get("/api/assets/{kind}")
async def api_assets(kind: str) -> JSONResponse:
    items = list_assets(kind)
    payload: Dict[str, Any] = {"items": items, "count": len(items)}
    if kind in ("keyposes", "trajectories"):
        order = KEYPOSE_CATEGORIES if kind == "keyposes" else TRAJECTORY_CATEGORIES
        fallback = "other" if kind == "keyposes" else "moderate"
        cats: Dict[str, int] = {}
        speakers: Dict[str, int] = {}
        freqs: Dict[str, int] = {}
        for item in items:
            cats[item["category"] or fallback] = cats.get(item["category"] or fallback, 0) + 1
            if item.get("speaker"):
                speakers[item["speaker"]] = speakers.get(item["speaker"], 0) + 1
            if item.get("frequency"):
                freqs[item["frequency"]] = freqs.get(item["frequency"], 0) + 1
        payload["categories"] = [{"id": c, "count": cats.get(c, 0)} for c in order if cats.get(c)] + [
            {"id": c, "count": n} for c, n in cats.items() if c not in order]
        payload["speakers"] = sorted(speakers.items())
        if kind == "trajectories":
            payload["frequencies"] = [{"id": f, "count": freqs[f]} for f in TRAJECTORY_FREQUENCIES if freqs.get(f)]
            payload["default"] = default_trajectory_id()
    return JSONResponse(payload)


@app.patch("/api/assets/{kind}/{asset_id}")
async def api_asset_patch(kind: str, asset_id: str, request: Request) -> JSONResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected")
    return JSONResponse(update_asset_overlay(kind, asset_id, body))


@app.get("/api/assets/trajectories/{asset_id}/path")
async def api_trajectory_path(asset_id: str) -> JSONResponse:
    base = trajectory_base(asset_id)
    trans, yaw = ts.integrate_deltas(base["delta"], base["init_pos"], base["init_yaw"])
    return JSONResponse({"id": asset_id, "path": ts.path_summary(trans, yaw, base["init_pos"])})


@app.post("/api/traj/preview")
async def api_traj_preview(request: Request) -> JSONResponse:
    body = await request.json()
    base_spec = body.get("base") or {}
    kind = str(base_spec.get("kind") or "still")
    frames = parse_int(base_spec.get("frames"), 0)
    if kind == "trajectory" and base_spec.get("id"):
        base = trajectory_base(str(base_spec["id"]))
    elif kind == "job" and base_spec.get("job_id"):
        base = job_base_trajectory(str(base_spec["job_id"]), parse_int(base_spec.get("chunk_index"), 0))
    else:
        n = max(2, frames or 300)
        base = {"delta": np.zeros((n, 4), np.float32), "init_pos": np.zeros(3, np.float32), "init_yaw": 0.0}
    delta = base["delta"]
    if frames and frames != delta.shape[0]:
        # loop / truncate to the requested length the same way the generator does
        reps = int(np.ceil(frames / delta.shape[0]))
        delta = np.concatenate([delta] * reps, axis=0)[:frames]
    try:
        segs = ts.normalize_script(body.get("script") or [], delta.shape[0])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    base_trans, base_yaw = ts.integrate_deltas(delta, base["init_pos"], base["init_yaw"])
    res = ts.synthesize(delta, segs, base["init_pos"], base["init_yaw"])
    return JSONResponse({
        "base": ts.path_summary(base_trans, base_yaw, base["init_pos"]),
        "result": ts.path_summary(res["trans"], res["yaw"], base["init_pos"]),
        "segments": [{**s.to_dict(), "description": ts.describe_segment(s)} for s in segs],
    })


# ---------------------------------------------------------------- jobs
@app.get("/api/jobs")
async def api_jobs(limit: int = 200) -> JSONResponse:
    items = [job_summary(s) for s in JOBS.list_statuses(limit)]
    return JSONResponse({"items": items, "version": JOBS.list_version, "current_job": JOBS.current_job})


@app.get("/api/jobs/version")
async def api_jobs_version() -> JSONResponse:
    active = any(s.get("state") in ACTIVE_STATES for s in JOBS.list_statuses())
    return JSONResponse({"version": JOBS.list_version, "active": active, "current_job": JOBS.current_job})


@app.post("/api/jobs")
async def api_create_job(request: Request) -> JSONResponse:
    form = await request.form()
    jid = make_job_id()
    directory = job_dir(jid)
    uploads = directory / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    audio_info = None
    audio = form.get("audio")
    if audio is not None and getattr(audio, "filename", ""):
        filename = safe_filename(str(audio.filename))
        dest = uploads / filename
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty audio upload")
        dest.write_bytes(content)
        audio_info = {"filename": filename, "path": str(dest.resolve()), "url": file_url(dest), "size": len(content),
                      "content_type": getattr(audio, "content_type", None)}
    elif form.get("audio_job_id"):
        src = job_dir(str(form.get("audio_job_id")))
        src_req = read_json(src / "request.json", {})
        src_audio = (src_req.get("audio") or {})
        src_path = Path(src_audio.get("path", ""))
        if not src_path.exists():
            raise HTTPException(status_code=400, detail="source job has no audio")
        dest = uploads / src_path.name
        shutil.copy2(src_path, dest)
        audio_info = {"filename": src_path.name, "path": str(dest.resolve()), "url": file_url(dest), "size": dest.stat().st_size,
                      "content_type": src_audio.get("content_type"), "copied_from": str(form.get("audio_job_id"))}
    if audio_info is None:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail="an audio file is required")

    identity = str(form.get("identity") or "") or list_assets("identities")[0]["id"]
    trajectory = str(form.get("trajectory") or "") or default_trajectory_id(identity)
    get_asset("identities", identity)
    get_asset("trajectories", trajectory)
    traj_script = validate_traj_script(form.get("traj_script"), None)
    payload = {
        "audio": audio_info,
        "identity": identity,
        "trajectory": trajectory,
        "traj_script": traj_script,
        "title": str(form.get("title") or "")[:120],
        "notes": str(form.get("notes") or "")[:2000],
        "render": parse_bool(form.get("render"), True) and RENDER_AVAILABLE,
        "guidance_scale": parse_float(form.get("guidance_scale"), 4.0),
        "skip_asr": parse_bool(form.get("skip_asr"), False),
        "language": str(form.get("language") or "English"),
        "source_job_id": "",
    }
    return JSONResponse(JOBS.create("generate", payload, jid))


@app.get("/api/jobs/{jid}")
async def api_job(jid: str) -> JSONResponse:
    return JSONResponse(job_detail(jid))


@app.patch("/api/jobs/{jid}")
async def api_job_patch(jid: str, request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(JOBS.patch(jid, body if isinstance(body, dict) else {}))


@app.post("/api/jobs/{jid}/cancel")
async def api_job_cancel(jid: str) -> JSONResponse:
    return JSONResponse(JOBS.cancel(jid))


@app.delete("/api/jobs/{jid}")
async def api_job_delete(jid: str, purge: int = 0) -> JSONResponse:
    if purge:
        JOBS.purge(jid)
        return JSONResponse({"job_id": jid, "deleted": True})
    return JSONResponse(JOBS.cancel(jid))


@app.post("/api/jobs/{jid}/edit")
async def api_job_edit(jid: str, request: Request) -> JSONResponse:
    source = job_detail(jid)
    if source["state"] != "succeeded":
        raise HTTPException(status_code=409, detail="source job has not succeeded")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected")
    total = source["total_frames"] or None
    keyposes = validate_keyposes(body.get("keyposes"), total)
    traj = validate_traj_script(body.get("traj_script"), total)
    if not keyposes and not traj:
        raise HTTPException(status_code=400, detail="nothing to apply: add a keypose or a trajectory segment")
    new_id = make_job_id()
    payload = {
        "source_job_id": jid,
        "source_job_dir": str(job_dir(jid).resolve()),
        "root_job_id": source["root_job_id"],
        "keyposes": keyposes,
        "traj_script": traj,
        "regen_windows": int(np.clip(parse_int(body.get("regen_windows"), 3), 1, 8)),
        "render": parse_bool(body.get("render"), True) and RENDER_AVAILABLE,
        "title": str(body.get("title") or "")[:120],
        "notes": str(body.get("notes") or "")[:2000],
    }
    return JSONResponse(JOBS.create("edit", payload, new_id))


@app.get("/api/jobs/{jid}/words")
async def api_job_words(jid: str) -> JSONResponse:
    detail = job_detail(jid)
    path = JOBS_DIR / detail["root_job_id"] / "features" / "words.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no word timings yet")
    data = read_json(path, {})
    data["job_id"] = detail["root_job_id"]
    return JSONResponse(data)


@app.post("/api/jobs/{jid}/transcribe")
async def api_job_transcribe(jid: str, request: Request) -> JSONResponse:
    detail = job_detail(jid)
    root = detail["root_job_id"]
    if not detail["audio_url"]:
        raise HTTPException(status_code=409, detail="job has no audio")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    body = body if isinstance(body, dict) else {}
    new_id = make_job_id()
    payload = {
        "target_job_id": root,
        "target_job_dir": str(job_dir(root).resolve()),
        "source_job_id": root,
        "language": str(body.get("language") or "English"),
        "force_asr": parse_bool(body.get("force_asr"), False),
        "title": f"words · {detail.get('title') or root}",
    }
    return JSONResponse(JOBS.create("transcribe", payload, new_id))


# ---------------------------------------------------------------- LLM edit assistant
@app.get("/api/agent/config")
async def api_agent_config() -> JSONResponse:
    return JSONResponse(agent_mod.public_config(agent_mod.load_config(AGENT_CONFIG_PATH)))


@app.put("/api/agent/config")
async def api_agent_config_put(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected")
    cfg = agent_mod.save_config(AGENT_CONFIG_PATH, body)
    return JSONResponse(agent_mod.public_config(cfg))


@app.post("/api/jobs/{jid}/agent")
async def api_job_agent(jid: str, request: Request) -> JSONResponse:
    """Ask the external LLM for keypose insertions. Returns suggestions only; no job is created."""
    detail = job_detail(jid)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    body = body if isinstance(body, dict) else {}

    cfg = agent_mod.load_config(AGENT_CONFIG_PATH)
    max_edits = int(np.clip(parse_int(body.get("max_edits"), cfg["max_edits"]), 1, 40))
    min_gap = int(np.clip(parse_int(body.get("min_gap_frames"), cfg["min_gap_frames"]), 0, 600))
    instruction = str(body.get("instruction") or "")[:4000]

    keyposes = list_assets("keyposes")
    total_frames = int(detail.get("total_frames") or 0)
    if not total_frames:
        raise HTTPException(status_code=409, detail="the job has no frame count yet (wait until it has finished)")

    # transcript: word timings when available, otherwise the per-chunk text
    words_path = JOBS_DIR / detail["root_job_id"] / "features" / "words.json"
    words_json = read_json(words_path, {}) if words_path.exists() else {}
    audio_path = None
    a16 = list((JOBS_DIR / detail["root_job_id"] / "audio").glob("*_16k.wav"))
    if a16:
        audio_path = a16[0]

    context, ctx_meta = agent_mod.build_context(
        words_json, detail.get("transcripts"), total_frames, FPS, audio_path=audio_path
    )
    system, user = agent_mod.build_prompt(
        agent_mod.build_catalog(keyposes), context, instruction, max_edits, min_gap, FPS
    )
    if parse_bool(body.get("dry_run"), False):
        return JSONResponse({"prompt_chars": len(system) + len(user), "context": ctx_meta,
                             "system": system, "user": user, "endpoint": agent_mod.endpoint_url(cfg)})

    try:
        reply = await asyncio.to_thread(agent_mod.call_llm, cfg, system, user)
        edits, notes, summary = agent_mod.parse_edits(
            reply["text"], {k["id"] for k in keyposes}, total_frames, FPS, max_edits, min_gap
        )
    except agent_mod.AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    by_id = {k["id"]: k for k in keyposes}
    for item in edits:
        src = by_id.get(item["keypose_id"], {})
        item["label"] = src.get("name") or item["keypose_id"]
        item["category"] = src.get("category") or ""
    return JSONResponse({
        "job_id": jid, "edits": edits, "notes": notes, "summary": summary,
        "context": ctx_meta, "model": reply.get("model"), "usage": reply.get("usage"),
        "protocol": reply.get("protocol"), "prompt_chars": len(system) + len(user),
    })


@app.get("/api/jobs/{jid}/logs")
async def api_job_logs(jid: str, offset: int = 0) -> JSONResponse:
    return JSONResponse(JOBS.logs(jid, offset))


@app.get("/api/jobs/{jid}/events")
async def api_job_events(jid: str, request: Request) -> StreamingResponse:
    path = JOBS.events_path(jid)

    async def stream() -> Iterable[str]:
        position = 0
        last_ping = time.monotonic()
        while True:
            if await request.is_disconnected():
                break
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(position)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        position = handle.tell()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield f"event: {event.get('event') or 'message'}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            status = JOBS.status(jid)
            if status.get("state") in TERMINAL_STATES:
                yield f"event: done\ndata: {json.dumps({'status': status}, ensure_ascii=False)}\n\n"
                break
            if time.monotonic() - last_ping > 15:
                yield ": ping\n\n"
                last_ping = time.monotonic()
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- files
@app.get("/api/files/{path:path}")
async def api_file(path: str) -> FileResponse:
    raw = unquote(path)
    candidate = Path(raw)
    resolved = candidate.expanduser().resolve() if candidate.is_absolute() else (ROOT / raw).resolve()
    if not any(is_relative_to(resolved, root) for root in allowed_file_roots()):
        raise HTTPException(status_code=403, detail="File is outside the job and asset folders")
    if resolved == AGENT_CONFIG_PATH.resolve() or resolved.suffix.lower() in {".ckpt", ".pt", ".pth"}:
        raise HTTPException(status_code=403, detail="Not served")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(resolved))
