"""Model loading, per-window conditions and job files shared by the generation and edit jobs.

A generation job directory holds

  features/speech.npz          rhythm / semantic / mel of the whole audio (30 fps)
  features/clip_text.npz       CLIP text embedding per chunk
  features/words.json          word timings (for the timeline and the LLM assistant)
  transcripts.json             ASR text per chunk
  raw/chunkNNN.npy             generated motion per chunk; root translation = the trajectory condition
  bigru/chunkNNN.npy           the same with the root translation predicted from the generated pose
  full_raw.npy, full_bigru.npy all chunks concatenated
  renders/chunks/chunkNNN_body.mp4

An edit job holds the chunks it changed (raw/, bigru/, renders/) plus its own full_*.npy; the
other chunks are taken from the closest ancestor job (see `resolve_chunks`).

A chunk file is a dict saved with np.save: motion [T, 268] = axis-angle pose (165), root
translation (3), FLAME expression (100), plus the conditions it was generated with.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
BODY_CHECKPOINT = CHECKPOINT_DIR / "dynacontalk_edit"   # editable body model
FACE_CHECKPOINT = CHECKPOINT_DIR / "dynacontalk_face"
TRAJECTORY_CHECKPOINT = CHECKPOINT_DIR / "trajectory_bigru" / "model.ckpt"
ASSETS_DIR = Path(os.environ.get("STUDIO_ASSET_ROOT") or ROOT / "assets")

FPS = 30
WINDOW = 64
OVERLAP = 8
STRIDE = WINDOW - OVERLAP
POSE_DIM = 165
TRANS_DIM = 3
EXPR_DIM = 100


# ---------------------------------------------------------------- job files

def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {} if default is None else default


class Status:
    """Progress reports to the job's status.json (merged into what the server wrote)."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def __call__(self, state, phase, progress, message="", **extra):
        print(f"[{phase}] {message}", flush=True)
        if self.path is None:
            return
        status = read_json(self.path)
        status.update({"state": state, "phase": phase, "progress": float(progress), "message": message,
                       "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra})
        write_json(self.path, status)


def report_failure(status_path, exc: BaseException) -> None:
    import traceback

    Status(status_path)("failed", "error", 1.0, str(exc), error_type=type(exc).__name__,
                        traceback=traceback.format_exc())


def load_chunk(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def chunk_name(index: int) -> str:
    return f"chunk{index:03d}"


def job_lineage(job_dir: Path) -> list[Path]:
    """[job_dir, parent, ..., root generation job], following request.json source_job_dir."""
    chain, seen = [], set()
    cur = Path(job_dir).resolve()
    while cur and cur not in seen and cur.exists():
        seen.add(cur)
        chain.append(cur)
        parent = read_json(cur / "request.json").get("source_job_dir")
        cur = Path(parent).resolve() if parent else None
    return chain


def resolve_chunks(job_dir: Path) -> list[dict]:
    """Chunk table of a job: frame range and the newest raw / bigru file of every chunk along the lineage."""
    lineage = job_lineage(job_dir)
    root = lineage[-1]
    rows = read_json(root / "manifest.json").get("chunks") or []
    chunks = []
    for row in rows:
        name = chunk_name(int(row["index"]))
        raw = next(d / "raw" / f"{name}.npy" for d in lineage if (d / "raw" / f"{name}.npy").exists())
        bigru = next(d / "bigru" / f"{name}.npy" for d in lineage if (d / "bigru" / f"{name}.npy").exists())
        chunks.append({"index": int(row["index"]), "name": name, "start": int(row["frame_start"]),
                       "end": int(row["frame_end"]), "raw": raw, "bigru": bigru})
    return chunks


def concatenate_chunks(paths: list[Path], out_path: Path, audio_path, extra: dict | None = None) -> None:
    """Concatenate chunk files, shifting each chunk's root translation to continue the previous one."""
    motions, first = [], None
    for path in paths:
        obj = load_chunk(path)
        first = first or obj
        motion = np.asarray(obj["motion"], dtype=np.float32).copy()
        if motions:
            motion[:, POSE_DIM:POSE_DIM + TRANS_DIM] += (
                motions[-1][-1, POSE_DIM:POSE_DIM + TRANS_DIM] - motion[0, POSE_DIM:POSE_DIM + TRANS_DIM])[None, :]
        motions.append(motion)
    out = dict(first)
    out.update({"motion": np.concatenate(motions, axis=0).astype(np.float32), "sample_key": "full",
                "audio_path": str(audio_path), "total_frames": int(sum(m.shape[0] for m in motions)),
                "chunk_files": [str(p) for p in paths], **(extra or {})})
    np.save(out_path, out)


# ---------------------------------------------------------------- models

def setup_torch() -> None:
    # full fp32, as the training features were extracted (TF32 shifts the HuBERT features)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Generator:
    """A released DynaConTalk model with its normalization statistics."""

    def __init__(self, checkpoint_dir: Path, device: str):
        import hydra
        from omegaconf import OmegaConf

        checkpoint_dir = Path(checkpoint_dir)
        cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
        cfg.data.data_dir = str(checkpoint_dir)
        cfg.paths.output_dir = str(checkpoint_dir)
        cfg.model.enable_fgd = False
        model = hydra.utils.instantiate(cfg.model)
        state = torch.load(checkpoint_dir / "model.ckpt", map_location="cpu", weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{checkpoint_dir}: checkpoint does not match its config "
                               f"(missing={len(missing)}, unexpected={len(unexpected)})")
        # frozen parameters also select PyTorch's fused Transformer inference kernels
        self.model = model.to(device).eval().requires_grad_(False)
        self.model.sample_scheduler.set_timesteps(int(model.hparams.step_num))
        self.device = device
        stats = np.load(checkpoint_dir / "stats.npz")
        self.mean, self.std = stats["motion_mean"], stats["motion_std"]
        self.shape_mean, self.shape_std = stats["shape_betas_mean"], stats["shape_betas_std"]
        conditioning = model.denoiser.conditioning_module
        self.activity_dim = int(conditioning.activity_dim)
        self.shape_dim = int(conditioning.shape_dim)
        self.uses_trajectory = bool(conditioning.use_trajectory_condition)
        self.target_group = str(model.hparams.target_group)

    def activity(self, activity) -> torch.Tensor:
        arr = np.asarray(activity, dtype=np.float32).reshape(-1)
        if arr.shape[0] < self.activity_dim:
            arr = np.pad(arr, (0, self.activity_dim - arr.shape[0]))
        return torch.from_numpy(arr[:self.activity_dim].astype(np.float32))

    def windows(self, features: dict, total: int, clip_text, shape_betas, trans=None, root_orient=None) -> list:
        """Per-window conditions for `total` frames: speech features, CLIP text, normalized body shape
        and (editable model) the root trajectory condition from the root translation / orientation."""
        n = 1 if total <= WINDOW else 1 + (total - WINDOW + STRIDE - 1) // STRIDE
        shape = np.asarray(shape_betas, dtype=np.float32).reshape(-1)[:self.shape_dim]
        shape = (shape - self.shape_mean[:self.shape_dim]) / (self.shape_std[:self.shape_dim] + 1e-8)
        shape = torch.from_numpy(shape.astype(np.float32)).float()
        coarse = None
        if self.uses_trajectory:
            trans = np.asarray(trans, dtype=np.float32)[:total]
            coarse = np.concatenate([trans - trans[:1], np.asarray(root_orient, dtype=np.float32)[:total, :3]], axis=-1)
        windows = []
        for i in range(n):
            start = i * STRIDE
            w = {k: torch.from_numpy(_slice_pad(features[k], start, WINDOW)).float() for k in ("rhythm", "semantic", "mel")}
            w["clip_text"] = torch.from_numpy(np.asarray(clip_text, dtype=np.float32)).float()
            if coarse is not None:
                coarse_t = torch.from_numpy(_slice_pad(coarse, start, WINDOW)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    w["trajectory_cond"] = self.model._trajectory_condition_from_coarse(coarse_t).squeeze(0).cpu()
            w["shape_betas"] = shape
            windows.append(w)
        return windows

    def normalize(self, wavelet: np.ndarray) -> np.ndarray:
        return ((wavelet - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def history(self, seed_wavelet: np.ndarray, total: int) -> torch.Tensor:
        """Normalized first-window history from the seed sequence (repeated if the clip is shorter)."""
        past = np.asarray(seed_wavelet, dtype=np.float32)[:total][:OVERLAP]
        if past.shape[0] < OVERLAP:
            past = np.concatenate([past, np.repeat(past[-1:], OVERLAP - past.shape[0], axis=0)], axis=0)
        return torch.from_numpy(self.normalize(past)).float().to(self.device)

    @torch.no_grad()
    def decode(self, wavelet_norm: np.ndarray) -> np.ndarray:
        """Normalized wavelet motion [T, D] -> motion [T, D / 4]."""
        wavelet = np.asarray(wavelet_norm, dtype=np.float32) * self.std + self.mean
        x = torch.from_numpy(wavelet).float().unsqueeze(0).to(self.device)
        return self.model._wavelet_to_motion(x).squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode(self, motion: np.ndarray) -> np.ndarray:
        """Motion [T, C] -> normalized wavelet motion [T, 4 C]."""
        x = torch.from_numpy(np.asarray(motion, dtype=np.float32)).float().unsqueeze(0).to(self.device)
        wavelet = self.model._motion_to_wavelet(x).squeeze(0).cpu().numpy()
        return self.normalize(wavelet)

    def sample(self, windows, total, activity, history, guidance_scale, edit=None) -> np.ndarray:
        """Normalized wavelet motion [total, D]."""
        out = self.model.sample_motion(windows, total_frames=total, window_size=WINDOW, guidance_scale=guidance_scale,
                                       activity=self.activity(activity), init_past_motion=history, edit=edit)
        return out.cpu().numpy()


def _slice_pad(arr, start: int, length: int) -> np.ndarray:
    """arr[start:start + length], padded by repeating the last frame (zeros if empty)."""
    arr = np.asarray(arr, dtype=np.float32)
    seg = arr[start:min(start + length, arr.shape[0])]
    if seg.shape[0] == 0:
        return np.zeros((length,) + arr.shape[1:], dtype=np.float32)
    if seg.shape[0] < length:
        seg = np.concatenate([seg, np.repeat(seg[-1:], length - seg.shape[0], axis=0)], axis=0)
    return seg.astype(np.float32)


def free_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loop_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    """Repeat a sequence along time until it has `length` frames."""
    arr = np.asarray(arr)
    if arr.shape[0] == length:
        return arr.copy()
    reps = int(math.ceil(length / float(arr.shape[0])))
    return np.concatenate([arr] * reps, axis=0)[:length].copy()
