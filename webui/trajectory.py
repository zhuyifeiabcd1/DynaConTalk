"""Root-translation BiGRU: predicts where the character walks from its generated body pose.

The body models are conditioned on a root trajectory but output only joint rotations; the
rendered root translation is predicted from the generated pose by this small network
(pelvis-relative joint positions, their velocities and the root yaw rate -> per-frame
translation relative to the first frame). Network: src/models/trajectory_bigru.py.
"""
from pathlib import Path

import numpy as np
import torch

from src.models.emage_evaltools.rotation_conversions import axis_angle_to_matrix
from src.models.trajectory_bigru import TrajectoryBiGRU

POSE_DIM = 165
TRANS_DIM = 3


def _time_derivative(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Central finite difference over time of [B, T, ...] (one-sided at the ends)."""
    out = torch.zeros_like(values)
    if values.shape[1] <= 1:
        return out
    if values.shape[1] == 2:
        out[:, 0] = out[:, 1] = (values[:, 1] - values[:, 0]) / dt
        return out
    out[:, 0] = (values[:, 1] - values[:, 0]) / dt
    out[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt)
    out[:, -1] = (values[:, -1] - values[:, -2]) / dt
    return out


def _angle_derivative(angles: torch.Tensor, dt: float) -> torch.Tensor:
    def diff(a, b):
        delta = a - b
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    out = torch.zeros_like(angles)
    if angles.shape[1] <= 1:
        return out
    if angles.shape[1] == 2:
        out[:, 0] = out[:, 1] = diff(angles[:, 1], angles[:, 0]) / dt
        return out
    out[:, 0] = diff(angles[:, 1], angles[:, 0]) / dt
    out[:, 1:-1] = diff(angles[:, 2:], angles[:, :-2]) / (2.0 * dt)
    out[:, -1] = diff(angles[:, -1], angles[:, -2]) / dt
    return out


class TrajectoryPredictor:
    def __init__(self, checkpoint: Path, device: str):
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        self.cfg = cfg
        self.device = device
        self.model = TrajectoryBiGRU(cfg["pose_dim"], cfg["hidden_size"], cfg["num_layers"], cfg["dropout"],
                                     cfg["patch_size"], cfg["cnn_dropout"], cfg["output_dim"])
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def features(self, poses_aa: np.ndarray) -> torch.Tensor:
        """[T, 165] axis-angle pose -> [1, T, F]: pelvis-centred joint positions rotated by the root
        orientation, their velocities, and the root yaw rate."""
        from src.models.emage_evaltools.motion_rep_transfer import get_motion_rep_tensor

        cfg, t = self.cfg, poses_aa.shape[0]
        motion_aa = torch.from_numpy(poses_aa).float().view(1, t, POSE_DIM).to(self.device)
        joints = get_motion_rep_tensor(motion_aa, pose_fps=cfg["pose_fps"], device=self.device)["position"]
        if cfg["center_pelvis"]:
            joints = joints - joints[:, :, 0:1, :]
        if cfg["use_root_orient"]:
            root_rot = axis_angle_to_matrix(motion_aa[:, :, :3])
            joints = torch.matmul(root_rot.unsqueeze(2), joints.unsqueeze(-1)).squeeze(-1)
        joints = joints[:, :, cfg["input_joint_indices"], :]
        dt = 1.0 / float(cfg["pose_fps"])
        parts = [joints.reshape(1, t, -1)]
        if cfg["include_joint_velocity"]:
            parts.append(_time_derivative(joints, dt).reshape(1, t, -1))
        if cfg["include_root_yaw_rate"]:
            fwd = axis_angle_to_matrix(motion_aa[:, :, :3])[..., 2]
            yaw = torch.atan2(fwd[..., 0], fwd[..., 2])
            parts.append(_angle_derivative(yaw, dt).unsqueeze(-1))
        return torch.cat(parts, dim=-1).cpu()

    @torch.no_grad()
    def rewrite(self, motion: np.ndarray) -> np.ndarray:
        """Predicted root translation [T, 3] for a [T, >=168] motion (pose + translation),
        starting at the motion's first translation."""
        poses_aa = motion[:, :POSE_DIM].astype(np.float32)
        feat = self.features(poses_aa)
        lengths = torch.tensor([poses_aa.shape[0]], dtype=torch.long, device=self.device)
        init = torch.from_numpy(motion[:1, POSE_DIM:POSE_DIM + TRANS_DIM].reshape(1, 1, 3).astype(np.float32))
        rel = self.model(feat.to(self.device), lengths=lengths)
        return (rel + init.float().to(self.device)).squeeze(0).cpu().numpy().astype(np.float32)
