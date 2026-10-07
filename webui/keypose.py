"""Keypose targets for editing: blend a pose into a motion around a frame and build the repaint mask."""
import json
from pathlib import Path

import numpy as np
import torch

from webui.rotations import (
    NUM_JOINTS,
    POSE_DIM_6D,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    slerp_quaternion,
)

BODY_PART_JOINTS = {
    "full_body": list(range(NUM_JOINTS)),
    "torso": [0, 1, 2, 3, 6, 9, 12, 15],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "both_arms": [13, 14, 16, 17, 18, 19, 20, 21],
    "hands": list(range(20, 22)) + list(range(25, 55)),
    "upper_body": [3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21] + list(range(25, 55)),
}

# wavelet bands of the 3-level transform, in the interleaved channel order
WAVELET_BANDS = {"ca3": 0, "cd3": 1, "cd2": 2, "cd1": 3}


def load_keypose(assets_dir: Path, keypose_id: str) -> tuple[dict, np.ndarray]:
    """Manifest record and [330] 6D target pose of a keypose asset."""
    root = Path(assets_dir) / "keyposes"
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("id") == keypose_id:
            with np.load(root / record["pose"]) as z:
                return record, np.asarray(z["pose_rot6d"], dtype=np.float32).reshape(POSE_DIM_6D)
    raise KeyError(f"keypose {keypose_id!r} not found in {root / 'manifest.jsonl'}")


def _part_channels(joints: list[int]) -> np.ndarray:
    return np.asarray([c for j in joints for c in range(j * 6, j * 6 + 6)], dtype=np.int64)


def _temporal_weights(length: int, frame: int, sigma: float, strength: float) -> np.ndarray:
    frame = max(0, min(int(frame), length - 1))
    sigma = max(float(sigma), 1e-6)
    t = np.arange(length, dtype=np.float32)
    weights = np.exp(-0.5 * ((t - float(frame)) / sigma) ** 2)
    return np.clip(weights * float(strength), 0.0, 1.0).astype(np.float32)


def apply_keyposes(base_6d: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Blend keyposes into a [T, 330] 6D motion by slerp with Gaussian-in-time weights.

    specs: {"frame", "part", "sigma", "strength", "target_pose": [330]}. Keyposes are applied in
    frame order, each starting from the motion already edited by the previous ones, so a row of
    keyposes is passed through one after another. Returns the edited motion and the per-channel
    weight mask (max over keyposes).
    """
    base = np.asarray(base_6d, dtype=np.float32)
    edited = base.copy()
    mask = np.zeros_like(base, dtype=np.float32)
    length = base.shape[0]
    for spec in sorted(specs, key=lambda s: int(s["frame"])):
        joints = BODY_PART_JOINTS[str(spec["part"])]
        channels = _part_channels(joints)
        target = np.asarray(spec["target_pose"], dtype=np.float32).reshape(NUM_JOINTS, 6)[joints]
        weights_np = _temporal_weights(length, int(spec["frame"]), float(spec["sigma"]), float(spec["strength"]))

        base_part = torch.from_numpy(edited[:, channels].reshape(length, len(joints), 6))
        target_part = torch.from_numpy(target.reshape(1, len(joints), 6)).expand(length, -1, -1)
        base_quat = matrix_to_quaternion(rotation_6d_to_matrix(base_part))
        target_quat = matrix_to_quaternion(rotation_6d_to_matrix(target_part))
        weights = torch.from_numpy(weights_np).view(length, 1, 1).to(base_quat)
        edited_part = matrix_to_rotation_6d(quaternion_to_matrix(slerp_quaternion(base_quat, target_quat, weights)))
        edited[:, channels] = edited_part.reshape(length, -1).cpu().numpy().astype(np.float32)
        mask[:, channels] = np.maximum(mask[:, channels], weights_np[:, None])
    return edited, mask


def wavelet_mask(motion_mask: np.ndarray, bands: list[str], levels: int = 3) -> np.ndarray:
    """[T, 330] motion-space mask -> [T, 330 * (levels + 1)] interleaved wavelet mask on the given bands."""
    mask = np.asarray(motion_mask, dtype=np.float32)
    band_mask = np.zeros(levels + 1, dtype=np.float32)
    for band in bands:
        band_mask[WAVELET_BANDS[band]] = 1.0
    return (mask[:, :, None] * band_mask[None, None, :]).reshape(mask.shape[0], mask.shape[1] * (levels + 1))
