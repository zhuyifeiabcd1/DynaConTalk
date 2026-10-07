"""Trajectory script primitives (numpy only).

A *trajectory script* is a list of segments applied on top of a base root-delta
sequence.  Root deltas follow the model's trajectory-condition convention
(`src/models/light_final.py::_root_trajectory_from_coarse`):

    delta[t] = [delta_yaw, dx_local, dz_local, y_offset]

* ``delta_yaw``  yaw change between frame t-1 and t (radians).  Positive turns the
  character to its left (counter-clockwise seen from above, y up).
* ``dx_local`` / ``dz_local``  root displacement between t-1 and t expressed in the
  heading frame of frame t-1.  ``+z`` is forward, ``+x`` is the character's left
  (SMPL-X body frame: x left, y up, z forward).
* ``y_offset``  root height relative to the first frame of the clip (metres).

The module is deliberately torch-free so the web server can use it for live
previews and the generation / edit jobs use the very same code.

Segment schema (all frames are clip-local, ``end`` exclusive)::

    {
      "id": "seg_1",            # optional, echoed back
      "start": 300, "end": 336, # frames
      "type": "forward",        # see PRIMITIVES
      "amount": 1.2,            # metres for moves, degrees for turns, metres for crouch/rise
      "steps": 2,               # optional: for forward/backward/strafe, amount = steps * STEP_LENGTH
      "mode": "replace",        # "replace" (default) fades the base motion out inside the
                                # segment, "add" stacks the primitive on top of it
      "hold": 0.4               # crouch/rise only: fraction of the segment spent at full depth
    }
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

FPS = 30
STEP_LENGTH = 0.6  # metres per walking step used when a segment gives ``steps``
BLEND_FRAMES = 8   # base motion fade in/out at segment borders (replace mode)

PRIMITIVES: dict[str, dict[str, Any]] = {
    "forward": {"channel": "dz", "sign": +1.0, "unit": "m", "default": 1.2, "label": "Walk forward"},
    "backward": {"channel": "dz", "sign": -1.0, "unit": "m", "default": 1.2, "label": "Step back"},
    "strafe_left": {"channel": "dx", "sign": +1.0, "unit": "m", "default": 0.6, "label": "Side-step left"},
    "strafe_right": {"channel": "dx", "sign": -1.0, "unit": "m", "default": 0.6, "label": "Side-step right"},
    "turn_left": {"channel": "yaw", "sign": +1.0, "unit": "deg", "default": 45.0, "label": "Turn left"},
    "turn_right": {"channel": "yaw", "sign": -1.0, "unit": "deg", "default": 45.0, "label": "Turn right"},
    "crouch": {"channel": "y", "sign": -1.0, "unit": "m", "default": 0.12, "label": "Crouch (dip)"},
    "rise": {"channel": "y", "sign": +1.0, "unit": "m", "default": 0.06, "label": "Rise (tip-toe)"},
    "hold": {"channel": None, "sign": 0.0, "unit": "", "default": 0.0, "label": "Stand still"},
}


# --------------------------------------------------------------------------- rotations
def rot_y(yaw: np.ndarray | float) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.clip(angle, 1e-8, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    a = angle[..., 0]
    ca, sa = np.cos(a), np.sin(a)
    one = 1.0 - ca
    m = np.empty(aa.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = ca + x * x * one
    m[..., 0, 1] = x * y * one - z * sa
    m[..., 0, 2] = x * z * one + y * sa
    m[..., 1, 0] = y * x * one + z * sa
    m[..., 1, 1] = ca + y * y * one
    m[..., 1, 2] = y * z * one - x * sa
    m[..., 2, 0] = z * x * one - y * sa
    m[..., 2, 1] = z * y * one + x * sa
    m[..., 2, 2] = ca + z * z * one
    return m


def matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = np.clip((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    axis = np.stack(
        [m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], axis=-1
    )
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    small = norm[..., 0] < 1e-8
    axis = axis / np.clip(norm, 1e-8, None)
    out = axis * angle[..., None]
    # near-zero rotation: axis undefined, return zeros
    out[small] = 0.0
    return out.astype(np.float32)


def yaw_from_matrix(m: np.ndarray) -> np.ndarray:
    """Heading of the body z-axis projected on the ground plane (matches the model)."""
    return np.arctan2(m[..., 0, 2], m[..., 2, 2])


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------- delta <-> world
def deltas_from_trans_root(trans: np.ndarray, root_aa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[T,3] world trans + [T,3] root axis-angle -> ([T,4] deltas, init_pos[3], init_yaw)."""
    trans = np.asarray(trans, dtype=np.float64)
    pos = trans - trans[:1]
    yaw = yaw_from_matrix(axis_angle_to_matrix(np.asarray(root_aa, dtype=np.float64)))
    T = pos.shape[0]
    delta = np.zeros((T, 4), dtype=np.float64)
    if T > 1:
        delta[1:, 0] = wrap_angle(yaw[1:] - yaw[:-1])
        dpos = pos[1:] - pos[:-1]
        yaw_prev = yaw[:-1]
        cy, sy = np.cos(yaw_prev), np.sin(yaw_prev)
        delta[1:, 1] = cy * dpos[:, 0] - sy * dpos[:, 2]
        delta[1:, 2] = sy * dpos[:, 0] + cy * dpos[:, 2]
    delta[:, 3] = pos[:, 1]
    return delta.astype(np.float32), trans[0].astype(np.float32), float(yaw[0])


def integrate_deltas(delta: np.ndarray, init_pos: np.ndarray, init_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """[T,4] deltas -> ([T,3] world trans, [T] yaw)."""
    delta = np.asarray(delta, dtype=np.float64)
    T = delta.shape[0]
    init_pos = np.asarray(init_pos, dtype=np.float64).reshape(3)
    yaw = float(init_yaw) + np.cumsum(delta[:, 0])
    yaw[0] = float(init_yaw)  # delta[0,0] is defined as 0 but be explicit
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[0] = init_pos
    cur = init_pos.copy()
    prev_yaw = float(init_yaw)
    for i in range(1, T):
        cy, sy = math.cos(prev_yaw), math.sin(prev_yaw)
        dx_local, dz_local = delta[i, 1], delta[i, 2]
        cur = cur.copy()
        cur[0] += cy * dx_local + sy * dz_local
        cur[2] += -sy * dx_local + cy * dz_local
        cur[1] = init_pos[1] + delta[i, 3]
        pos[i] = cur
        prev_yaw = yaw[i]
    return pos.astype(np.float32), yaw.astype(np.float32)


def root_orient_with_yaw(root_aa: np.ndarray, yaw_new: np.ndarray) -> np.ndarray:
    """Replace the heading of ``root_aa`` by ``yaw_new`` while keeping pitch/roll."""
    m = axis_angle_to_matrix(np.asarray(root_aa, dtype=np.float64))
    yaw_old = yaw_from_matrix(m)
    fix = rot_y(np.asarray(yaw_new, dtype=np.float64) - yaw_old)
    return matrix_to_axis_angle(fix @ m)


# --------------------------------------------------------------------------- profiles
def bump_profile(n: int) -> np.ndarray:
    """Raised-cosine velocity profile of length n that sums to exactly 1."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    s = (np.arange(n, dtype=np.float64) + 0.5) / n
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * s))
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def dip_profile(n: int, hold: float = 0.4) -> np.ndarray:
    """0 -> 1 -> 0 envelope: smooth attack, plateau of ``hold`` fraction, smooth release."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    hold = float(np.clip(hold, 0.0, 0.9))
    ramp = max(1, int(round(n * (1.0 - hold) / 2.0)))
    t = np.arange(n, dtype=np.float64)
    attack = smoothstep(t / ramp)
    release = smoothstep((n - 1 - t) / ramp)
    return np.minimum(attack, release)


def edge_fade(n: int, blend: int) -> np.ndarray:
    """1 inside, fading to 0 at both borders over ``blend`` frames (for base fade-out)."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    blend = int(max(0, min(blend, n // 2)))
    w = np.ones(n, dtype=np.float64)
    if blend > 0:
        ramp = (np.arange(blend, dtype=np.float64) + 1.0) / (blend + 1.0)
        w[:blend] = ramp
        w[n - blend:] = ramp[::-1]
    return w


# --------------------------------------------------------------------------- script
@dataclass
class Segment:
    start: int
    end: int
    type: str
    amount: float
    mode: str = "replace"
    hold: float = 0.4
    id: str = ""
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "start": int(self.start),
            "end": int(self.end),
            "type": self.type,
            "amount": float(self.amount),
            "mode": self.mode,
            "hold": float(self.hold),
            "label": self.label,
        }
        d.update(self.extra)
        return d


def normalize_segment(raw: dict[str, Any], total_frames: int | None = None) -> Segment:
    if not isinstance(raw, dict):
        raise ValueError("segment must be an object")
    stype = str(raw.get("type", "")).strip().lower()
    if stype not in PRIMITIVES:
        raise ValueError(f"unknown primitive '{stype}', expected one of {sorted(PRIMITIVES)}")
    start = int(round(float(raw.get("start", 0))))
    end = int(round(float(raw.get("end", start))))
    if total_frames is not None:
        start = max(0, min(start, max(0, total_frames - 1)))
        end = max(0, min(end, total_frames))
    if end <= start:
        raise ValueError(f"segment '{raw.get('id', '')}' has empty range [{start}, {end})")
    prim = PRIMITIVES[stype]
    amount = raw.get("amount", None)
    steps = raw.get("steps", None)
    if steps is not None and prim["channel"] in ("dx", "dz"):
        amount = float(steps) * STEP_LENGTH
    if amount is None or amount == "":
        amount = prim["default"]
    amount = abs(float(amount))
    mode = str(raw.get("mode", "replace")).strip().lower()
    if mode not in ("replace", "add"):
        mode = "replace"
    hold = float(raw.get("hold", 0.4))
    extra = {k: v for k, v in raw.items() if k not in {"id", "start", "end", "type", "amount", "mode", "hold", "label", "steps"}}
    if steps is not None:
        extra["steps"] = float(steps)
    return Segment(
        start=start,
        end=end,
        type=stype,
        amount=amount,
        mode=mode,
        hold=hold,
        id=str(raw.get("id", "")),
        label=str(raw.get("label", "")),
        extra=extra,
    )


def normalize_script(script: Iterable[dict[str, Any]] | None, total_frames: int | None = None) -> list[Segment]:
    segments = [normalize_segment(s, total_frames) for s in (script or [])]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def apply_script(base_delta: np.ndarray, segments: Iterable[Segment], blend: int = BLEND_FRAMES) -> np.ndarray:
    """Return a new [T,4] delta sequence with the script applied."""
    base = np.asarray(base_delta, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 4:
        raise ValueError(f"base_delta must be [T,4], got {base.shape}")
    T = base.shape[0]
    out = base.copy()
    for seg in segments:
        s, e = max(0, seg.start), min(T, seg.end)
        n = e - s
        if n <= 0:
            continue
        prim = PRIMITIVES[seg.type]
        channel = prim["channel"]
        signed = prim["sign"] * seg.amount
        if seg.mode == "replace" and channel != "y":
            keep = 1.0 - edge_fade(n, blend)  # base contribution fades out inside the segment
            out[s:e, 0:3] = base[s:e, 0:3] * keep[:, None]
        if channel == "dz":
            out[s:e, 2] += signed * bump_profile(n)
        elif channel == "dx":
            out[s:e, 1] += signed * bump_profile(n)
        elif channel == "yaw":
            out[s:e, 0] += math.radians(signed) * bump_profile(n)
        elif channel == "y":
            out[s:e, 3] = base[s:e, 3] + signed * dip_profile(n, seg.hold)
        # "hold": base already faded out, nothing to add
    return out.astype(np.float32)


def synthesize(
    base_delta: np.ndarray,
    script: Iterable[dict[str, Any]] | Iterable[Segment],
    init_pos: np.ndarray | None = None,
    init_yaw: float = 0.0,
    blend: int = BLEND_FRAMES,
) -> dict[str, Any]:
    """Apply a script and integrate.  Returns delta, trans, yaw and the normalized segments."""
    base = np.asarray(base_delta, dtype=np.float32)
    T = base.shape[0]
    segs = [s if isinstance(s, Segment) else normalize_segment(s, T) for s in script]
    segs.sort(key=lambda s: (s.start, s.end))
    delta = apply_script(base, segs, blend=blend)
    if init_pos is None:
        init_pos = np.zeros(3, dtype=np.float32)
    trans, yaw = integrate_deltas(delta, init_pos, init_yaw)
    return {"delta": delta, "trans": trans, "yaw": yaw, "segments": segs}


def path_summary(trans: np.ndarray, yaw: np.ndarray, init_pos: np.ndarray | None = None, max_points: int = 600, fps: int = FPS) -> dict[str, Any]:
    """Downsampled top-down path for UI previews (relative to the first frame)."""
    trans = np.asarray(trans, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    T = trans.shape[0]
    origin = trans[0] if init_pos is None else np.asarray(init_pos, dtype=np.float64)
    rel = trans - origin[None, :]
    step = max(1, int(math.ceil(T / max_points)))
    idx = np.arange(0, T, step)
    if idx[-1] != T - 1:
        idx = np.append(idx, T - 1)
    speed = np.zeros(T)
    if T > 1:
        speed[1:] = np.linalg.norm(trans[1:, [0, 2]] - trans[:-1, [0, 2]], axis=1) * fps
    return {
        "frames": int(T),
        "fps": int(fps),
        "index": idx.astype(int).tolist(),
        "x": np.round(rel[idx, 0], 4).tolist(),
        "z": np.round(rel[idx, 2], 4).tolist(),
        "y": np.round(rel[idx, 1], 4).tolist(),
        "yaw": np.round(yaw[idx], 4).tolist(),
        "speed": np.round(speed[idx], 3).tolist(),
        "bbox": {
            "x": [float(rel[:, 0].min()), float(rel[:, 0].max())],
            "z": [float(rel[:, 2].min()), float(rel[:, 2].max())],
        },
    }


def describe_segment(seg: Segment) -> str:
    prim = PRIMITIVES[seg.type]
    if prim["channel"] is None:
        return prim["label"]
    if prim["unit"] == "deg":
        return f"{prim['label']} {seg.amount:.0f}°"
    if "steps" in seg.extra:
        return f"{prim['label']} {seg.extra['steps']:g} step(s)"
    return f"{prim['label']} {seg.amount:.2f} m"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":  # pragma: no cover
    T = 300
    base = np.zeros((T, 4), dtype=np.float32)
    script = [
        {"id": "a", "start": 30, "end": 66, "type": "forward", "steps": 2},
        {"id": "b", "start": 100, "end": 130, "type": "turn_left", "amount": 90},
        {"id": "c", "start": 150, "end": 186, "type": "forward", "amount": 1.0},
        {"id": "d", "start": 200, "end": 260, "type": "crouch", "amount": 0.12},
    ]
    res = synthesize(base, script, init_pos=np.zeros(3), init_yaw=0.0)
    tr, yw, dl = res["trans"], res["yaw"], res["delta"]
    print("after forward 2 steps: z=%.3f (expect 1.200)" % tr[66, 2])
    print("after turn: yaw=%.3f rad (expect 1.571)" % yw[130])
    print("after forward along new heading: x=%.3f (expect +1.0), z=%.3f (expect 1.2)" % (tr[186, 0], tr[186, 2]))
    print("crouch min y=%.3f (expect -0.120)" % tr[200:260, 1].min())
    # round trip through world -> deltas
    root_aa = root_orient_with_yaw(np.zeros((T, 3), dtype=np.float32), yw)
    d2, p0, y0 = deltas_from_trans_root(tr, root_aa)
    print("roundtrip max |delta err| = %.2e" % np.abs(d2 - dl).max())
    print(path_summary(tr, yw)["bbox"])
