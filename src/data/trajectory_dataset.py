"""BEAT2 sequences for the root-translation BiGRU (src/models/trajectory_bigru.py).

Each sample is a whole sequence: trajectory features [T, F] computed from the body pose and
the root translation relative to the first frame [T, 3]. The features are precomputed by
src/tools/preprocess_trajectory.py.
"""
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.emage_evaltools.rotation_conversions import axis_angle_to_matrix

# joints of the trajectory features: pelvis, hips, spine, knees, ankles, feet, neck, shoulders
DEFAULT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17]


def _time_derivative(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Central finite difference over time of [T, ...] (one-sided at the ends)."""
    length = int(values.shape[0])
    out = torch.zeros_like(values)
    if length <= 1:
        return out
    if length == 2:
        out[0] = out[1] = (values[1] - values[0]) / dt
        return out
    out[0] = (values[1] - values[0]) / dt
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def _angle_derivative(angles: torch.Tensor, dt: float) -> torch.Tensor:
    """Time derivative of wrapped angles [T]."""
    def diff(a, b):
        delta = a - b
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    length = int(angles.shape[0])
    out = torch.zeros_like(angles)
    if length <= 1:
        return out
    if length == 2:
        out[0] = out[1] = diff(angles[1], angles[0]) / dt
        return out
    out[0] = diff(angles[1], angles[0]) / dt
    out[1:-1] = diff(angles[2:], angles[:-2]) / (2.0 * dt)
    out[-1] = diff(angles[-1], angles[-2]) / dt
    return out


def trajectory_features(poses: np.ndarray, joint_indices: Optional[List[int]] = None, pose_fps: int = 30,
                        center_pelvis: bool = True, use_root_orient: bool = True,
                        include_joint_velocity: bool = True, include_root_yaw_rate: bool = True) -> np.ndarray:
    """SMPL-X axis-angle poses [T, 165] -> trajectory features [T, F]: joint positions (pelvis-centred,
    rotated by the root orientation), their velocities and the root yaw rate."""
    from src.models.emage_evaltools.motion_rep_transfer import get_motion_rep_numpy  # loads SMPL-X

    poses = np.asarray(poses, dtype=np.float32)
    t = poses.shape[0]
    joints = torch.from_numpy(get_motion_rep_numpy(poses, pose_fps=pose_fps, device="cpu")["position"]).float()
    if center_pelvis:
        joints = joints - joints[:, 0:1, :]
    root_rot = axis_angle_to_matrix(torch.from_numpy(poses[:, :3]).float())
    if use_root_orient:
        joints = torch.matmul(root_rot.unsqueeze(1), joints.unsqueeze(-1)).squeeze(-1)
    joints = joints[:, DEFAULT_JOINTS if joint_indices is None else list(joint_indices), :]

    dt = 1.0 / float(pose_fps)
    parts = [joints.reshape(t, -1)]
    if include_joint_velocity:
        parts.append(_time_derivative(joints, dt).reshape(t, -1))
    if include_root_yaw_rate:
        forward = root_rot[:, :, 2]  # local +Z in world coordinates
        yaw = torch.atan2(forward[:, 0], forward[:, 2])
        parts.append(_angle_derivative(yaw, dt).unsqueeze(-1))
    return torch.cat(parts, dim=-1).numpy()


class TrajectoryDataset(Dataset):
    """Whole sequences of a preprocessed file {key: {"input_feat": [T, F], "trans": [T, 3]}}."""

    def __init__(self, npy_path: str):
        super().__init__()
        data = np.load(npy_path, allow_pickle=True).item()
        self.samples = []
        for key, sample in data.items():
            feat = torch.from_numpy(np.asarray(sample["input_feat"], dtype=np.float32)).float()
            trans = np.asarray(sample["trans"], dtype=np.float32)
            length = min(int(feat.shape[0]), len(trans))
            self.samples.append((key, feat[:length], trans[:length]))
        if not self.samples:
            raise RuntimeError(f"no sequences in {npy_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, feat, trans = self.samples[idx]
        return {
            "input_feat": feat,
            "trans_rel": torch.from_numpy(trans - trans[0:1]).float(),
            "length": torch.tensor(feat.shape[0], dtype=torch.long),
            "sample_id": key,
        }


def collate_sequences(batch):
    """Zero-pad a batch of sequences to the longest one."""
    max_len = max(int(item["length"]) for item in batch)
    feat = torch.zeros(len(batch), max_len, batch[0]["input_feat"].shape[-1])
    trans_rel = torch.zeros(len(batch), max_len, 3)
    for i, item in enumerate(batch):
        n = int(item["length"])
        feat[i, :n] = item["input_feat"]
        trans_rel[i, :n] = item["trans_rel"]
    return {"input_feat": feat, "trans_rel": trans_rel, "length": torch.stack([item["length"] for item in batch]),
            "sample_ids": [item["sample_id"] for item in batch]}


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, train_file: str = "trajectory_train.npy", val_file: str = "trajectory_val.npy",
                 batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_file, self.val_file = train_file, val_file
        self.batch_size, self.num_workers, self.pin_memory = int(batch_size), int(num_workers), bool(pin_memory)
        self.train_dataset = self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = TrajectoryDataset(str(self.data_dir / self.train_file))
            self.val_dataset = TrajectoryDataset(str(self.data_dir / self.val_file))

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_sequences)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)
