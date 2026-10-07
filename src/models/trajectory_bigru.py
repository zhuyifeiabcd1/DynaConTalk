"""Root-translation BiGRU: predicts the root translation of a motion from its body pose.

The body models output joint rotations; the Studio predicts the root translation from the
generated pose with this network: per-frame features (pelvis-centred joint positions rotated
by the root orientation, their velocities and the root yaw rate; src/data/trajectory_dataset.py)
-> translation relative to the first frame.

Train: bash scripts/train_trajectory.sh; export for the Studio: src/tools/export_trajectory.py.
"""
from typing import List, Optional

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class TrajectoryModule(L.LightningModule):
    """Training of TrajectoryBiGRU: L1 on the translation, its velocity, acceleration and end point."""

    def __init__(
        self,
        pose_dim: int = 91,
        hidden_size: int = 384,
        num_layers: int = 3,
        dropout: float = 0.12,
        patch_size: int = 16,
        cnn_dropout: float = 0.05,
        output_dim: int = 3,
        input_joint_indices: Optional[List[int]] = None,
        pose_fps: int = 30,
        center_pelvis: bool = True,
        use_root_orient: bool = True,
        include_joint_velocity: bool = True,
        include_root_yaw_rate: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler=None,
        noise_std: float = 0.003,
        loss_pos_weight: float = 1.0,
        loss_vel_weight: float = 0.5,
        loss_acc_weight: float = 0.2,
        loss_end_weight: float = 0.3,
    ):
        super().__init__()
        # the feature settings are stored with the weights so that the Studio computes the same features
        self.save_hyperparameters(ignore=["lr_scheduler"])
        self._lr_scheduler = lr_scheduler
        self.net = TrajectoryBiGRU(pose_dim, hidden_size, num_layers, dropout, patch_size, cnn_dropout, output_dim)

    def forward(self, input_feat, lengths):
        return self.net(input_feat, lengths)

    @staticmethod
    def _masked_l1(pred, target, mask):
        m = mask.unsqueeze(-1).to(dtype=pred.dtype)
        return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)

    def _shared_step(self, batch, stage):
        input_feat, trans_rel, lengths = batch["input_feat"], batch["trans_rel"], batch["length"]
        batch_size, max_len, _ = input_feat.shape
        valid = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

        # small input noise: the Studio feeds generated, not captured, poses
        if self.training and float(self.hparams.noise_std) > 0.0:
            noise = torch.randn_like(input_feat) * float(self.hparams.noise_std)
            input_feat = input_feat + noise * valid.unsqueeze(-1).to(dtype=input_feat.dtype)

        pred = self.forward(input_feat, lengths)
        pos_loss = self._masked_l1(pred, trans_rel, valid)
        pred_vel, gt_vel = pred[:, 1:] - pred[:, :-1], trans_rel[:, 1:] - trans_rel[:, :-1]
        vel_loss = self._masked_l1(pred_vel, gt_vel, valid[:, 1:])
        pred_acc, gt_acc = pred_vel[:, 1:] - pred_vel[:, :-1], gt_vel[:, 1:] - gt_vel[:, :-1]
        acc_loss = self._masked_l1(pred_acc, gt_acc, valid[:, 2:])
        rows = torch.arange(batch_size, device=lengths.device)
        last = (lengths - 1).clamp(min=0)
        end_loss = torch.abs(pred[rows, last] - trans_rel[rows, last]).sum(dim=-1).mean()

        hp = self.hparams
        loss = (float(hp.loss_pos_weight) * pos_loss + float(hp.loss_vel_weight) * vel_loss
                + float(hp.loss_acc_weight) * acc_loss + float(hp.loss_end_weight) * end_loss)

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in (("pos", pos_loss), ("vel", vel_loss), ("acc", acc_loss), ("end", end_loss)):
            self.log(f"{stage}/{name}_loss", value, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr),
                                      weight_decay=float(self.hparams.weight_decay))
        if self._lr_scheduler is None:
            return optimizer
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": self._lr_scheduler(optimizer=optimizer), "interval": "epoch"}}

    def studio_config(self) -> dict:
        """Network and feature settings in the form webui/trajectory.py reads."""
        hp = self.hparams
        return {
            "pose_dim": int(hp.pose_dim), "hidden_size": int(hp.hidden_size), "num_layers": int(hp.num_layers),
            "dropout": float(hp.dropout), "patch_size": int(hp.patch_size), "cnn_dropout": float(hp.cnn_dropout),
            "output_dim": int(hp.output_dim), "input_joint_indices": [int(i) for i in hp.input_joint_indices],
            "pose_fps": int(hp.pose_fps), "center_pelvis": bool(hp.center_pelvis),
            "use_root_orient": bool(hp.use_root_orient), "include_joint_velocity": bool(hp.include_joint_velocity),
            "include_root_yaw_rate": bool(hp.include_root_yaw_rate),
        }
