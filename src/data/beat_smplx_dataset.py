"""BEAT2 windows for training and validation.

Reads beat2_{train,val,test}.npy written by src/tools/preprocess_beat2.py. Each item is a
fixed-length window of one sequence: the normalized wavelet motion of the target group
(body or face), the speech features, the CLIP transcript embedding of the window, the
speaker id, the normalized body shape and the root trajectory (relative translation and
root orientation, used by the editable model).
"""
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

POSE_DIM = 330  # 55 joints x 6D
TRANS_DIM = 3


class BeatSMPLXDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        split: str,
        window_size: int = 64,
        stride: int = 56,
        wavelet_levels: int = 3,
        target_group: str = "body",
        activity_dim: int = 1,
    ):
        super().__init__()
        if target_group not in {"body", "face"}:
            raise ValueError(f"target_group must be 'body' or 'face', got {target_group!r}")
        self.processed_dir = Path(processed_dir)
        self.window_size = window_size
        self.wavelet_levels = int(wavelet_levels)
        self.target_group = target_group
        self.activity_dim = int(activity_dim)

        self.data: Dict[str, Dict] = np.load(self.processed_dir / f"beat2_{split}.npy", allow_pickle=True).item()
        self.sample_index = []
        for key, sample in self.data.items():
            length = sample.get("length", len(sample["poses"]))
            for start in range(0, length - window_size + 1, stride):
                self.sample_index.append((key, start))
        if not self.sample_index:
            raise RuntimeError(f"no windows of {window_size} frames in {split}")

        mean = self._select_target(np.load(self.processed_dir / "wavelet_mean.npy").astype(np.float32))
        std = self._select_target(np.load(self.processed_dir / "wavelet_std.npy").astype(np.float32))
        std[std == 0] = 1e-6
        self.motion_mean, self.motion_std = mean, std
        self.motion_mean_torch = torch.from_numpy(mean)
        self.motion_std_torch = torch.from_numpy(std)
        shape_mean = np.load(self.processed_dir / "shape_betas_mean.npy").astype(np.float32)
        shape_std = np.load(self.processed_dir / "shape_betas_std.npy").astype(np.float32)
        shape_std[shape_std == 0] = 1e-6
        self.shape_mean_torch = torch.from_numpy(shape_mean)
        self.shape_std_torch = torch.from_numpy(shape_std)

    def __len__(self) -> int:
        return len(self.sample_index)

    def _select_target(self, arr: np.ndarray) -> np.ndarray:
        """Interleaved wavelet [..., 433 * (L+1)] = [rotations 330, root translation 3, face 100]
        -> body [..., 330 * (L+1)] or face [..., 100 * (L+1)]."""
        l1 = self.wavelet_levels + 1
        x = arr.reshape(*arr.shape[:-1], arr.shape[-1] // l1, l1)
        x = x[..., :POSE_DIM, :] if self.target_group == "body" else x[..., POSE_DIM + TRANS_DIM:, :]
        return x.reshape(*arr.shape[:-1], x.shape[-2] * l1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key, start = self.sample_index[idx]
        sample = self.data[key]
        end = start + self.window_size
        poses = sample["poses"][start:end]
        trans = sample["trans"][start:end]
        trans_relative = trans - trans[0:1].copy()

        wavelet = self._select_target(sample["wavelet"][start:end].astype(np.float32))
        motion = (torch.from_numpy(wavelet).float() - self.motion_mean_torch) / self.motion_std_torch

        # the CLIP embedding of the preprocessed window that starts at (or nearest to) this window
        starts = np.asarray(sample["window_starts"]).astype(np.int32)
        hit = np.where(starts == start)[0]
        win_idx = int(hit[0]) if hit.size > 0 else int(np.argmin(np.abs(starts - start)))
        clip_text = np.asarray(sample["clip_text_windows"])[win_idx]

        activity = np.asarray(sample["activity"], dtype=np.float32).reshape(-1)[: self.activity_dim]
        shape = torch.from_numpy(np.asarray(sample["shape_betas"], dtype=np.float32).reshape(-1)[: self.shape_mean_torch.numel()])
        return {
            "audio_features": {
                name: torch.from_numpy(sample["audio_features"][name][start:end]).float()
                for name in ("rhythm", "semantic", "mel")
            },
            "motion": motion.float(),
            # root translation relative to the window's first frame + root orientation
            "coarse_trajectory": torch.from_numpy(np.concatenate([trans_relative[:, :3], poses[:, :3]], axis=-1)).float(),
            "motion_len": torch.tensor(self.window_size, dtype=torch.long),
            "activity": torch.from_numpy(activity).float(),
            "clip_text": torch.from_numpy(np.asarray(clip_text).astype(np.float32)).float(),
            "shape_betas": ((shape.float() - self.shape_mean_torch) / self.shape_std_torch).float(),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    out = {key: torch.stack([b[key] for b in batch]) for key in batch[0] if key != "audio_features"}
    out["audio_features"] = {
        name: torch.stack([b["audio_features"][name] for b in batch]) for name in batch[0]["audio_features"]
    }
    return out


class BeatSMPLXDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 24,
        num_workers: int = 4,
        pin_memory: bool = True,
        window_size: int = 64,
        overlap_size: int = 8,
        train_stride: int = 10,
        eval_stride: int = 56,
        motion_dim: int = 1320,
        wavelet_levels: int = 3,
        target_group: str = "body",
        activity_dim: int = 1,
    ):
        """
        window_size / overlap_size   window length and history frames (shared with the model)
        train_stride / eval_stride   window start spacing in the training / validation split
        motion_dim                   model input size, checked against the data: 1320 body, 400 face
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.train_stride = train_stride
        self.eval_stride = eval_stride
        self.motion_dim = motion_dim
        self.wavelet_levels = wavelet_levels
        self.target_group = target_group
        self.activity_dim = activity_dim

    def _dataset(self, split: str, stride: int) -> BeatSMPLXDataset:
        dataset = BeatSMPLXDataset(
            self.data_dir, split, window_size=self.window_size, stride=stride,
            wavelet_levels=self.wavelet_levels, target_group=self.target_group, activity_dim=self.activity_dim,
        )
        if dataset.motion_mean.shape[0] != self.motion_dim:
            raise RuntimeError(f"data has {dataset.motion_mean.shape[0]} motion channels, config says {self.motion_dim}")
        return dataset

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = self._dataset("train", self.train_stride)
            self.val_dataset = self._dataset("val", self.eval_stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, collate_fn=collate_fn)
