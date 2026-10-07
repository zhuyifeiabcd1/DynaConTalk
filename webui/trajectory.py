"""Root-translation BiGRU: predicts where the character walks from its generated body pose.

The body models are conditioned on a root trajectory but output only joint rotations; the
rendered root translation is predicted from the generated pose by this small network
(pelvis-relative joint positions, their velocities and the root yaw rate -> per-frame
translation relative to the first frame).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.emage_evaltools.rotation_conversions import axis_angle_to_matrix

POSE_DIM = 165
TRANS_DIM = 3


class TrajectoryBiGRU(nn.Module):
    """Frame features -> patch tokens (conv) -> BiGRU -> frame-level translation (transposed conv)."""

    def __init__(self, pose_dim, hidden_size=256, num_layers=2, dropout=0.1, patch_size=16, cnn_dropout=0.05,
                 output_dim=3):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(pose_dim, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=self.patch_size, stride=self.patch_size),
            nn.SiLU(),
        )
        self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout if num_layers > 1 else 0.0, bidirectional=True, batch_first=True)
        dec_hidden = max(64, hidden_size // 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=self.patch_size, stride=self.patch_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, dec_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(dec_hidden, output_dim, kernel_size=1),
        )

    def forward(self, input_feat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, T, F] features, [B] lengths -> [B, T, 3] translation relative to the first frame."""
        frame_len = input_feat.shape[1]
        pad_len = (self.patch_size - frame_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            input_feat = F.pad(input_feat, (0, 0, 0, pad_len))
        patch_tokens = self.patch_embed(input_feat.transpose(1, 2)).transpose(1, 2)
        patch_lengths = ((lengths + self.patch_size - 1) // self.patch_size).clamp(min=1, max=patch_tokens.shape[1])
        packed = pack_padded_sequence(patch_tokens, lengths=patch_lengths.detach().cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=patch_tokens.shape[1])
        return self.decoder(encoded.transpose(1, 2)).transpose(1, 2)[:, :frame_len, :]


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
