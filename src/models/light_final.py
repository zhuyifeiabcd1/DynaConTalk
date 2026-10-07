"""DynaConTalk training module.

Diffusion on 64-frame windows of wavelet-domain motion (x0 prediction): the first 8 frames of
a window are the motion history, re-noised at every denoising step at inference
(inpainting), the rest is generated. Long sequences are generated window by window
(stride 56) with dead blending at the window joins.
"""
import math
import os

import lightning.pytorch as L
import torch
import torch.nn.functional as F

from .emage_evaltools import FGD, rotation_6d_to_axis_angle
from .emage_evaltools import rotation_conversions as rc
from .emage_evaltools.motion_rep_transfer import get_motion_rep_tensor
from .nets.ema import EMAModel
from .wavelet import MotionWaveletISWT, MotionWaveletSWT

POSE_DIM = 330  # 55 SMPL-X joints x 6D rotation
SPEECH_KEYS = ("rhythm", "semantic", "mel")

# SMPL-X joints of the body parts used for keypose conditioning
KEYPOSE_PART_JOINTS = {
    "torso": [0, 1, 2, 3, 6, 9, 12, 15],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "both_arms": [13, 14, 16, 17, 18, 19, 20, 21],
    "hands": list(range(20, 22)) + list(range(25, 55)),
}


class LightMotionGeneration(L.LightningModule):
    def __init__(
        self,
        denoiser,
        noise_scheduler,
        sample_scheduler,
        optimizer,
        lr_scheduler=None,
        ema=None,
        guidance_scale: float = 4.0,
        step_num: int = 10,
        window_size: int = 64,
        overlap_size: int = 8,
        blend_frames: int = 4,
        wavelet_levels: int = 3,
        wavelet_name: str = "db6",
        target_group: str = "body",
        enable_fgd: bool = True,
        fgd_fallback_value: float = 1e6,
        uncond_prob: float = 0.2,
        partial_cond_prob: float = 0.4,
        partial_drop_max_modalities: int = 4,
        loss_weight_body: float = 2.0,
        use_loss_3d: bool = True,
        use_loss_3d_vel: bool = True,
        lambda_3d: float = 0.2,
        lambda_vel: float = 0.5,
        smoothness_loss_weight: float = 1.0,
        use_foot_skate_loss: bool = False,
        lambda_foot_skate: float = 0.0,
        foot_joint_indices=(7, 8, 10, 11),
        foot_contact_vel_threshold: float = 0.01,
        foot_contact_height_threshold: float = 0.05,
        use_keypose_condition: bool = False,
        keypose_no_prob: float = 1.0,
        keypose_single_prob: float = 0.0,
        keypose_multi_prob: float = 0.0,
        keypose_max_multi: int = 3,
        keypose_train_sigma: float = 6.0,
        keypose_train_strength: float = 1.0,
        keypose_train_parts=("torso", "left_arm", "right_arm", "both_arms", "hands"),
        lambda_keypose: float = 0.0,
        lambda_gate_sparsity: float = 0.0,
        save_every_n_epochs: int = 100,
        ckpt_path: str = None,
    ):
        """
        denoiser / noise_scheduler / sample_scheduler
                               network (nets/light_final.LightT2M), DDPM training schedule,
                               UniPC sampling schedule (step_num steps, CFG guidance_scale)
        window_size / overlap_size / blend_frames
                               window length, history frames per window, dead-blend frames
        target_group           "body" or "face"
        uncond_prob            probability of the null condition per training sample (CFG)
        partial_cond_prob      probability of dropping 1-3 of the speech / transcript inputs
        loss_weight_body       weight of the diffusion loss channels (normalized out per sample)
        use_loss_3d / use_loss_3d_vel / lambda_3d / lambda_vel
                               SMPL-X joint position / velocity losses
        smoothness_loss_weight continuity between the last history frame and the first generated one
        use_foot_skate_loss / lambda_foot_skate / foot_*
                               foot sliding under the conditioned root trajectory (editable model)
        use_keypose_condition / keypose_* / lambda_keypose
                               random sparse keypose targets during training (editable model)
        lambda_gate_sparsity   L1 penalty on the DGN v2 residual gates
        save_every_n_epochs    extra checkpoint every n epochs (written to ckpt_path)
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["denoiser"])
        self.denoiser = denoiser
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_scheduler
        self.sample_scheduler.set_timesteps(step_num)
        self.ema_denoiser = None
        if ema is not None and ema.use_ema:
            self.ema_denoiser = EMAModel(self.denoiser, decay=ema.ema_decay)
            self.ema_denoiser.set(self.denoiser)

        self.fgd_evaluator = None
        if enable_fgd:
            self.fgd_evaluator = FGD(download_path=os.path.join(os.path.dirname(__file__), "emage_evaltools"))

        self.motion_dim = self.denoiser.motion_dim
        self.target_group = target_group
        self.wavelet_levels = wavelet_levels
        self.wavelet_channels = self.motion_dim // (wavelet_levels + 1)
        self.wavelet_iswt = MotionWaveletISWT(self.wavelet_channels, levels=wavelet_levels, wavelet=wavelet_name)
        self.motion_mean = None  # normalization statistics, set from the data module
        self.motion_std = None
        num_params = sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad)
        print("number of trainable parameters: %.3fM" % (num_params / 1_000_000))

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer([p for p in self.denoiser.parameters() if p.requires_grad])
        if self.hparams.lr_scheduler is None:
            return optimizer
        lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

    def _set_motion_stats(self):
        if self.motion_mean is None or self.motion_std is None:
            dm = self.trainer.datamodule
            dataset = getattr(dm, "train_dataset", None) or getattr(dm, "val_dataset", None)
            self.motion_mean = torch.from_numpy(dataset.motion_mean).float().to(self.device)
            self.motion_std = torch.from_numpy(dataset.motion_std).float().to(self.device)

    def _wavelet_to_motion(self, wavelet_coeff: torch.Tensor) -> torch.Tensor:
        """[B, T, C * (levels + 1)] interleaved wavelet coefficients -> [B, T, C] motion."""
        return self.wavelet_iswt(wavelet_coeff.permute(0, 2, 1)).permute(0, 2, 1)

    def _motion_to_wavelet(self, motion: torch.Tensor) -> torch.Tensor:
        """[B, T, C] motion -> [B, T, C * (levels + 1)] interleaved wavelet coefficients."""
        swt = MotionWaveletSWT(self.wavelet_channels, levels=self.wavelet_levels, wavelet=self.hparams.wavelet_name)
        swt = swt.to(device=motion.device, dtype=motion.dtype)
        coeff = swt(motion.permute(0, 2, 1).contiguous())  # [B, C * (L+1), T], band-major
        b, d, t = coeff.shape
        l1 = self.wavelet_levels + 1
        coeff = coeff.view(b, l1, d // l1, t).permute(0, 2, 1, 3).contiguous().view(b, d, t)
        return coeff.permute(0, 2, 1).contiguous()

    # ------------------------------------------------------------------ editable-model conditions

    def _keypose_part_indices(self, part: str, device: torch.device) -> torch.Tensor:
        """Interleaved wavelet channels of the 6D rotations of a body part."""
        l1 = self.wavelet_levels + 1
        indices = []
        for joint_idx in KEYPOSE_PART_JOINTS[part]:
            for channel in range(joint_idx * 6, joint_idx * 6 + 6):
                indices.extend(range(channel * l1, channel * l1 + l1))
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _build_random_keypose_condition(self, target_wavelet: torch.Tensor, overlap_size: int):
        """Sample sparse keypose targets (Gaussian-in-time masks over random body parts)."""
        if not self.training or not self.hparams.use_keypose_condition:
            return None, None, torch.tensor(0.0, device=target_wavelet.device)
        B, T, _ = target_wavelet.shape
        hp = self.hparams
        total_prob = max(hp.keypose_no_prob + hp.keypose_single_prob + hp.keypose_multi_prob, 1e-8)
        no_prob = hp.keypose_no_prob / total_prob
        single_prob = hp.keypose_single_prob / total_prob
        max_multi = max(2, int(hp.keypose_max_multi))
        sigma = max(float(hp.keypose_train_sigma), 1e-6)
        parts = list(hp.keypose_train_parts)

        mask = torch.zeros_like(target_wavelet)
        t_axis = torch.arange(T, device=target_wavelet.device, dtype=target_wavelet.dtype)
        min_frame = min(max(int(overlap_size), 0), T - 1)
        active = torch.zeros(B, device=target_wavelet.device, dtype=target_wavelet.dtype)
        for b in range(B):
            r = torch.rand((), device=target_wavelet.device).item()
            if r < no_prob:
                continue
            if r < no_prob + single_prob:
                num_keyposes = 1
            else:
                num_keyposes = int(torch.randint(2, max_multi + 1, (1,), device=target_wavelet.device).item())
            active[b] = 1.0
            for _ in range(num_keyposes):
                frame = int(torch.randint(min_frame, T, (1,), device=target_wavelet.device).item())
                part = parts[int(torch.randint(0, len(parts), (1,), device=target_wavelet.device).item())]
                channels = self._keypose_part_indices(part, target_wavelet.device)
                weights = torch.exp(-0.5 * ((t_axis - float(frame)) / sigma) ** 2)
                weights = (weights * float(hp.keypose_train_strength)).clamp(0.0, 1.0)
                mask[b, :, channels] = torch.maximum(mask[b, :, channels], weights[:, None])

        hint = torch.zeros_like(target_wavelet)
        if mask.max() > 0:
            hint = target_wavelet.detach() * mask
        return hint, mask, active.mean()

    @staticmethod
    def _angle_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    def _root_trajectory_from_coarse(self, coarse: torch.Tensor):
        """[B, T, 6] (root translation relative to frame 0, root orientation) ->
        per-frame [delta_yaw, dx_local, dz_local, height], positions and yaw."""
        pos = coarse[..., :3]
        rot = rc.axis_angle_to_matrix(coarse[..., 3:6])
        yaw = torch.atan2(rot[..., 0, 2], rot[..., 2, 2])
        delta_yaw = torch.zeros_like(yaw)
        if yaw.shape[1] > 1:
            delta_yaw[:, 1:] = self._angle_difference(yaw[:, 1:], yaw[:, :-1])
        delta_pos = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            delta_pos[:, 1:] = pos[:, 1:] - pos[:, :-1]
        yaw_prev = torch.cat([yaw[:, :1], yaw[:, :-1]], dim=1)
        cy = torch.cos(yaw_prev)
        sy = torch.sin(yaw_prev)
        dx_world = delta_pos[..., 0]
        dz_world = delta_pos[..., 2]
        dx_local = cy * dx_world - sy * dz_world
        dz_local = sy * dx_world + cy * dz_world
        return torch.stack([delta_yaw, dx_local, dz_local, pos[..., 1]], dim=-1), pos, yaw

    def _trajectory_condition_from_coarse(self, coarse):
        """Velocity-style root trajectory condition (no absolute start position)."""
        if coarse is None:
            return None
        return self._root_trajectory_from_coarse(coarse)[0]

    @staticmethod
    def _apply_root_yaw_translation(joints, root_pos, yaw):
        """Map root-local joints to world space using yaw and translation."""
        cy = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        sy = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
        x, y, z = joints[..., 0:1], joints[..., 1:2], joints[..., 2:3]
        world = torch.cat([cy * x + sy * z, y, -sy * x + cy * z], dim=-1)
        return world + root_pos.unsqueeze(2)

    def _compute_foot_skate_loss(self, pred_local_joints, gt_local_joints, coarse):
        """Horizontal foot speed of the prediction on frames where the ground-truth foot is in contact,
        with both placed in the world by the conditioned root trajectory."""
        zero = pred_local_joints.new_tensor(0.0)
        if coarse is None or pred_local_joints.shape[1] < 2:
            return zero
        foot_idx = [int(i) for i in self.hparams.foot_joint_indices]
        _, root_pos, root_yaw = self._root_trajectory_from_coarse(coarse)
        if root_pos.shape[1] != pred_local_joints.shape[1]:
            return zero
        pred_feet = self._apply_root_yaw_translation(pred_local_joints[:, :, foot_idx, :], root_pos, root_yaw)
        with torch.no_grad():
            gt_feet = self._apply_root_yaw_translation(gt_local_joints[:, :, foot_idx, :], root_pos, root_yaw)
            gt_foot_speed = torch.linalg.norm((gt_feet[:, 1:] - gt_feet[:, :-1])[..., [0, 2]], dim=-1)
            contact_mask = gt_foot_speed < float(self.hparams.foot_contact_vel_threshold)
            height_threshold = float(self.hparams.foot_contact_height_threshold)
            if height_threshold > 0.0:
                ground = gt_feet[..., 1].amin(dim=(1, 2), keepdim=True)
                contact_mask = contact_mask & (gt_feet[:, 1:, :, 1] <= ground + height_threshold)
        if not contact_mask.any():
            return zero
        pred_foot_speed = torch.linalg.norm((pred_feet[:, 1:] - pred_feet[:, :-1])[..., [0, 2]], dim=-1)
        return pred_foot_speed[contact_mask].mean()

    # ------------------------------------------------------------------ training

    def on_train_start(self):
        self._set_motion_stats()

    def _step_network(self, batch, batch_idx):
        """Losses on one random window per sequence."""
        hp = self.hparams
        motion, length = batch["motion"], batch["motion_len"]
        speech = batch["audio_features"]
        activity = batch["activity"]
        clip_text = batch.get("clip_text")
        shape_betas = batch.get("shape_betas")
        coarse_trajectory = batch.get("coarse_trajectory")
        window_size, overlap_size = hp.window_size, hp.overlap_size

        valid = length >= window_size
        if not valid.any():
            return {"loss": torch.tensor(0.0, device=motion.device, requires_grad=True)}
        motion, length = motion[valid], length[valid]
        speech = {key: val[valid] for key, val in speech.items()}
        activity = activity[valid]
        if clip_text is not None:
            clip_text = clip_text[valid]
        if shape_betas is not None:
            shape_betas = shape_betas[valid]
        if coarse_trajectory is not None:
            coarse_trajectory = coarse_trajectory[valid]
        batch_size = motion.size(0)

        starts = []
        for i in range(batch_size):
            max_start = max(0, length[i].item() - window_size)
            starts.append(torch.randint(0, max_start + 1, (1,), device=motion.device).item())
        window_motion = torch.stack([motion[i, s:s + window_size] for i, s in enumerate(starts)], dim=0)
        speech = {
            key: torch.stack([val[i, s:s + window_size] for i, s in enumerate(starts)], dim=0)
            for key, val in speech.items()
        }
        if coarse_trajectory is not None:
            coarse_trajectory = torch.stack(
                [coarse_trajectory[i, s:s + window_size] for i, s in enumerate(starts)], dim=0
            )
        trajectory_cond = self._trajectory_condition_from_coarse(coarse_trajectory)
        keypose_hint, keypose_mask, keypose_active_ratio = self._build_random_keypose_condition(
            window_motion, overlap_size=overlap_size
        )

        # classifier-free guidance: samples with the learned null condition
        uncond_prob = hp.uncond_prob
        if self.training and uncond_prob > 0:
            drop_mask = torch.rand(batch_size, device=motion.device) < uncond_prob
        else:
            drop_mask = torch.zeros(batch_size, device=motion.device, dtype=torch.bool)

        # partial condition dropout: on other samples, drop 1-3 of the speech / transcript inputs
        if self.training and hp.partial_cond_prob > 0.0:
            partial_prob = min(1.0, hp.partial_cond_prob / max(1e-8, 1.0 - float(uncond_prob)))
            partial_mask = (torch.rand(batch_size, device=motion.device) < partial_prob) & ~drop_mask
            modality_specs = [("rhythm", 0.8), ("semantic", 0.6), ("mel", 0.8)]
            if clip_text is not None:
                modality_specs.append(("clip_text", 1.0))
            num_modalities = len(modality_specs)
            if partial_mask.any():
                modality_weights = torch.tensor([w for _, w in modality_specs], device=motion.device, dtype=torch.float32)
                drop_selection = torch.zeros(batch_size, num_modalities, dtype=torch.bool, device=motion.device)
                max_drop = min(int(hp.partial_drop_max_modalities), 4, num_modalities - 1)
                k_probs = torch.tensor([0.45, 0.30, 0.20, 0.05], device=motion.device, dtype=torch.float32)[:max_drop]
                k_probs = k_probs / k_probs.sum()
                for row in torch.nonzero(partial_mask, as_tuple=False).squeeze(1).tolist():
                    k = int(torch.multinomial(k_probs, num_samples=1, replacement=True).item() + 1)
                    drop_selection[row, torch.multinomial(modality_weights, num_samples=k, replacement=False)] = True

                def drop(x, sample_drop):
                    return torch.where(sample_drop.view(batch_size, *([1] * (x.dim() - 1))), torch.zeros_like(x), x)

                for idx, (name, _) in enumerate(modality_specs):
                    if name == "clip_text":
                        clip_text = drop(clip_text, drop_selection[:, idx])
                    else:
                        speech[name] = drop(speech[name], drop_selection[:, idx])

        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=motion.device).long()
        noise = torch.randn_like(window_motion, device=motion.device)
        noisy_window = self.noise_scheduler.add_noise(window_motion, noise, timestep)
        padding_mask = torch.ones(batch_size, window_size, dtype=torch.bool, device=motion.device)
        output = self.denoiser(
            noisy_window,
            padding_mask,
            timestep,
            speech=speech,
            activity=activity,
            cond_drop_mask=drop_mask if (self.training and uncond_prob > 0) else None,
            clip_text=clip_text,
            shape_betas=shape_betas,
            trajectory_cond=trajectory_cond,
            keypose_hint=keypose_hint,
            keypose_mask=keypose_mask,
        )

        cond_cache = self.denoiser.pop_condition_cache()
        modality_gates = cond_cache.get("modality_gates")
        loss_gate_sparsity = torch.tensor(0.0, device=motion.device)
        if modality_gates is not None and hp.lambda_gate_sparsity > 0.0:
            loss_gate_sparsity = modality_gates.mean()

        # diffusion loss over the whole window (x0 prediction), per-channel weighted MSE
        diff = (output - window_motion) ** 2
        weights = torch.full((diff.shape[-1],), float(hp.loss_weight_body), device=diff.device).view(1, 1, -1)
        loss_samp = ((diff * weights).sum(dim=(1, 2)) / (weights.sum() * diff.shape[1] + 1e-8)).mean()

        loss_keypose = torch.tensor(0.0, device=motion.device)
        if keypose_mask is not None and keypose_mask.max() > 0:
            kp_mask = keypose_mask.to(device=motion.device, dtype=output.dtype)
            loss_keypose = (((output - window_motion) ** 2) * kp_mask).sum() / (kp_mask.sum() + 1e-8)

        # motion-space losses
        if self.motion_mean is None or self.motion_std is None:
            self._set_motion_stats()
        pred_geom = self._wavelet_to_motion(output * self.motion_std + self.motion_mean)
        gt_geom = self._wavelet_to_motion(window_motion * self.motion_std + self.motion_mean)

        loss_smo = torch.tensor(0.0, device=motion.device)
        if hp.smoothness_loss_weight > 0 and overlap_size > 0:
            # the first generated frame should continue the last history frame
            pos_diff = pred_geom[:, overlap_size, :POSE_DIM] - gt_geom[:, overlap_size - 1, :POSE_DIM]
            loss_smo = F.mse_loss(pos_diff, torch.zeros_like(pos_diff), reduction="mean")

        loss_geo_xyz = torch.tensor(0.0, device=motion.device)
        loss_geo_xyz_vel = torch.tensor(0.0, device=motion.device)
        loss_foot_skate = torch.tensor(0.0, device=motion.device)
        if hp.use_loss_3d:
            pred_aa = rotation_6d_to_axis_angle(pred_geom[..., :POSE_DIM].reshape(batch_size, -1, 55, 6))
            gt_aa = rotation_6d_to_axis_angle(gt_geom[..., :POSE_DIM].reshape(batch_size, -1, 55, 6))
            pred_rep = get_motion_rep_tensor(pred_aa.reshape(batch_size, -1, 55 * 3), pose_fps=30, device=motion.device)
            pred_xyz = pred_rep["position"]
            with torch.no_grad():
                gt_rep = get_motion_rep_tensor(gt_aa.reshape(batch_size, -1, 55 * 3), pose_fps=30, device=motion.device)
                gt_xyz = gt_rep["position"]
            loss_geo_xyz = ((pred_xyz - gt_xyz) ** 2).mean(dim=(1, 2, 3)).mean()
            if hp.use_loss_3d_vel and pred_xyz.shape[1] > 1:
                loss_geo_xyz_vel = ((pred_rep["velocity"] - gt_rep["velocity"]) ** 2).mean(dim=(1, 2, 3)).mean()
            if hp.use_foot_skate_loss:
                loss_foot_skate = self._compute_foot_skate_loss(pred_xyz, gt_xyz, coarse_trajectory)

        loss = (
            loss_samp
            + hp.lambda_3d * loss_geo_xyz
            + hp.lambda_vel * loss_geo_xyz_vel
            + hp.smoothness_loss_weight * loss_smo
            + float(hp.lambda_foot_skate) * loss_foot_skate
            + float(hp.lambda_keypose) * loss_keypose
            + float(hp.lambda_gate_sparsity) * loss_gate_sparsity
        )
        losses = {
            "loss": loss,
            "loss_data": loss_samp,
            "loss_geo_xyz": loss_geo_xyz,
            "loss_geo_xyz_vel": loss_geo_xyz_vel,
            "loss_smo": loss_smo,
            "loss_foot_skate": loss_foot_skate,
            "loss_keypose": loss_keypose,
            "keypose_active_ratio": keypose_active_ratio,
            "loss_gate_sparsity": loss_gate_sparsity,
        }
        # DGN v2 gate statistics: mean opening, variation over time, variation across samples
        if modality_gates is not None:
            gates = modality_gates.detach()
            for gate_idx, gate_name in enumerate(cond_cache["modality_gate_names"]):
                losses[f"dgn_gate_{gate_name}_mean"] = gates[..., gate_idx].mean()
                losses[f"dgn_gate_{gate_name}_tstd"] = gates[..., gate_idx].std(dim=1).mean()
                if gates.shape[0] > 1:
                    losses[f"dgn_gate_{gate_name}_sstd"] = gates[..., gate_idx].mean(dim=1).std()
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._step_network(batch, batch_idx)
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=False)
        for key in ("loss_foot_skate", "loss_keypose", "keypose_active_ratio", "loss_gate_sparsity"):
            if key in losses:
                self.log(f"train/{key}", losses[key], prog_bar=False, on_step=True, on_epoch=False)
        for key, val in losses.items():
            if key.startswith("dgn_gate_"):
                self.log(f"train/{key}", val, prog_bar=False, on_step=True, on_epoch=False)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_denoiser is not None:
            if self.global_step <= self.hparams.ema.ema_start:
                self.ema_denoiser.set(self.denoiser)
            else:
                self.ema_denoiser.update(self.denoiser)

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.save_every_n_epochs == 0:
            self.trainer.save_checkpoint(os.path.join(self.hparams.ckpt_path, f"epoch-{self.current_epoch}.ckpt"))

    # ------------------------------------------------------------------ validation

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return
        self._evaluate_windows(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = {
            "val/FGD": float(self.hparams.fgd_fallback_value),
            "val/MSE": 0.0,
            "val/PCK": 0.0,
            "val/Diversity": 0.0,
            "val/FaceMSE": 0.0,
        }
        if self.fgd_evaluator is not None:
            try:
                fgd_score = float(self.fgd_evaluator.compute())
                if not math.isfinite(fgd_score):
                    raise ValueError(f"FGD is non-finite: {fgd_score}")
                metrics["val/FGD"] = fgd_score
                if self.trainer.is_global_zero:
                    print(f"[Val] FGD: {fgd_score:.4f}")
            except Exception as e:
                if self.trainer.is_global_zero:
                    print(f"Warning: failed to compute FGD: {e}")
        for name, key in (("MSE", "_val_mse"), ("PCK", "_val_pck"), ("Diversity", "_val_div"), ("FaceMSE", "_val_face")):
            count = getattr(self, f"{key}_count", 0)
            if count > 0:
                metrics[f"val/{name}"] = getattr(self, f"{key}_sum") / count
        metrics.update({"epoch": self.trainer.current_epoch, "step": self.global_step})
        self.log_dict(metrics, sync_dist=True)

    def _evaluate_windows(self, batch, batch_idx):
        """Validation on one random window per sequence, history frames from the ground truth.
        Body: FGD (EMAGE AESKConv), MSE / PCK / Diversity on axis-angle poses; face: FaceMSE."""
        if batch_idx == 0:
            if self.fgd_evaluator is not None:
                self.fgd_evaluator.reset()
            for key in ("_val_mse", "_val_pck", "_val_div", "_val_face"):
                setattr(self, f"{key}_sum", 0.0)
                setattr(self, f"{key}_count", 0)

        window_size, overlap_size = self.hparams.window_size, self.hparams.overlap_size
        motion, length = batch["motion"], batch["motion_len"]
        valid = length >= window_size
        if not valid.any():
            return
        motion, length = motion[valid], length[valid]
        speech = {k: v[valid] for k, v in batch["audio_features"].items()}
        activity = batch["activity"][valid]
        B = motion.size(0)

        starts = []
        for i in range(B):
            max_start = max(0, length[i].item() - window_size)
            starts.append(torch.randint(0, max_start + 1, (1,), device=motion.device).item())
        gt_windows = torch.stack([motion[i, s:s + window_size] for i, s in enumerate(starts)])
        speech = {k: torch.stack([v[i, s:s + window_size] for i, s in enumerate(starts)]) for k, v in speech.items()}
        conditions = {}
        if batch.get("clip_text") is not None:
            conditions["clip_text"] = batch["clip_text"][valid]
        if batch.get("shape_betas") is not None:
            conditions["shape_betas"] = batch["shape_betas"][valid]
        if batch.get("coarse_trajectory") is not None:
            coarse = batch["coarse_trajectory"][valid]
            conditions["trajectory_cond"] = self._trajectory_condition_from_coarse(
                torch.stack([coarse[i, s:s + window_size] for i, s in enumerate(starts)])
            )

        pred_windows = self._sample_windows(
            speech, conditions, activity, gt_windows[:, :overlap_size, :], self.hparams.guidance_scale, window_size, overlap_size
        )

        if self.motion_mean is None or self.motion_std is None:
            self._set_motion_stats()
        pred_denorm = self._wavelet_to_motion(pred_windows * self.motion_std + self.motion_mean)
        gt_denorm = self._wavelet_to_motion(gt_windows * self.motion_std + self.motion_mean)

        if self.target_group == "face":
            face_mse = F.mse_loss(pred_denorm, gt_denorm, reduction="mean")
            self._val_face_sum += face_mse.item() * B
            self._val_face_count += B
            self._val_mse_sum += face_mse.item() * B
            self._val_mse_count += B
            return

        pred_aa = rotation_6d_to_axis_angle(pred_denorm[:, :, :POSE_DIM].reshape(B, window_size, 55, 6))
        gt_aa = rotation_6d_to_axis_angle(gt_denorm[:, :, :POSE_DIM].reshape(B, window_size, 55, 6))
        diff_square = (pred_aa - gt_aa) ** 2
        self._val_mse_sum += diff_square.mean().item() * B
        self._val_mse_count += B
        self._val_pck_sum += (diff_square.sum(dim=-1).sqrt() < 0.5).float().mean().item() * B
        self._val_pck_count += B

        # Diversity: mean pairwise L1 distance within blocks of up to 50 windows
        B_div = min(50, B)
        if B_div >= 2:
            pred_aa_cpu = pred_aa.detach().cpu()
            for idx in range(B // B_div):
                block = pred_aa_cpu[idx * B_div : (idx + 1) * B_div]
                div_val = 0.0
                for ii in range(B_div):
                    dif = block[ii + 1 :] - block[ii]
                    if dif.numel() == 0:
                        continue
                    div_val += dif.abs().mean(dim=(1, 2, 3)).sum().item()
                div_val = div_val * 2 / (B_div * (B_div - 1))
                self._val_div_sum += div_val * B_div
                self._val_div_count += B_div

        if self.fgd_evaluator is not None:
            pred_poses = pred_denorm[:, :, :POSE_DIM]
            gt_poses = gt_denorm[:, :, :POSE_DIM]
            if torch.isfinite(pred_poses).all() and torch.isfinite(gt_poses).all():
                self.fgd_evaluator.update(pred_poses.float(), gt_poses.float())
            elif self.trainer.is_global_zero:
                print("[Val] Skip FGD update: NaN/Inf in pose tensors.")

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def _sample_windows(self, speech, conditions, activity, clean_past, guidance_scale, window_size, overlap_size,
                        repaint=None):
        """Sample a batch of windows with classifier-free guidance.

        speech / conditions / activity are batched [B, ...]; clean_past [B, overlap, D] or None.
        When given, the history frames are re-noised to the current noise level at every step
        and set to the clean history at the end.

        repaint (keypose editing, editable model): {"base", "target", "mask": [B, W, D], "start"}.
        Denoising starts from the base motion noised to the level `start` of the schedule, the
        masked keypose target is re-noised into the sample at every step, and target and mask
        are also given to the network as the keypose condition.
        """
        B = activity.shape[0]
        D = self.motion_dim
        if repaint is not None:
            conditions = {**conditions, "keypose_hint": repaint["target"], "keypose_mask": repaint["mask"]}
        speech = {k: torch.cat([v, v], dim=0) for k, v in speech.items()}
        conditions = {k: torch.cat([v, v], dim=0) for k, v in conditions.items()}
        activity = torch.cat([activity, activity], dim=0)
        # second half of the batch: the learned null condition
        cond_drop_mask = torch.cat(
            [torch.zeros(B, device=self.device, dtype=torch.bool), torch.ones(B, device=self.device, dtype=torch.bool)],
            dim=0,
        )
        denoiser = self.ema_denoiser.model if self.ema_denoiser is not None else self.denoiser
        inject_past = clean_past is not None and overlap_size > 0

        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        timesteps = self.sample_scheduler.timesteps
        if repaint is not None:
            start = max(0.0, min(1.0, float(repaint["start"])))
            timesteps = timesteps[min(len(timesteps) - 1, max(0, int((1.0 - start) * len(timesteps)))):]
            noise_base = torch.randn_like(repaint["base"])
            pred_motion = self.noise_scheduler.add_noise(repaint["base"], noise_base, timesteps[0].to(self.device))
            noise_key = torch.randn_like(repaint["target"])
        else:
            pred_motion = torch.randn(B, window_size, D, device=self.device) * self.sample_scheduler.init_noise_sigma
        padding_mask = torch.ones(B, window_size, dtype=torch.bool, device=self.device)
        for t in timesteps:
            t_gpu = t.to(self.device)
            if inject_past:
                noise_past = torch.randn(B, overlap_size, D, device=self.device)
                pred_motion[:, :overlap_size, :] = self.noise_scheduler.add_noise(clean_past, noise_past, t_gpu)
            if repaint is not None:
                noisy_key = self.noise_scheduler.add_noise(repaint["target"], noise_key, t_gpu)
                pred_motion = pred_motion * (1.0 - repaint["mask"]) + noisy_key * repaint["mask"]
            output = denoiser(
                pred_motion.repeat([2, 1, 1]),
                padding_mask.repeat([2, 1]),
                t_gpu.repeat([2 * B]),
                speech=speech,
                activity=activity,
                cond_drop_mask=cond_drop_mask,
                **conditions,
            )
            cond_x0, uncond_x0 = output.chunk(2)
            model_output = uncond_x0 + guidance_scale * (cond_x0 - uncond_x0)
            pred_motion = self.sample_scheduler.step(model_output, t, pred_motion).prev_sample.float()
        if inject_past:
            pred_motion[:, :overlap_size, :] = clean_past
        return pred_motion

    def _dead_blend(self, prev_motion, curr_motion, blend_frames=4):
        """Append curr_motion to prev_motion, cross-fading its first frames with a linear
        extrapolation of prev_motion."""
        if blend_frames <= 0 or prev_motion.shape[0] < 2:
            return torch.cat([prev_motion, curr_motion], dim=0)
        blend_frames = min(blend_frames, prev_motion.shape[0], curr_motion.shape[0])
        prev_last = prev_motion[-1]
        prev_velocity = prev_motion[-1] - prev_motion[-2]
        extrapolated = torch.stack([prev_last + prev_velocity * (i + 1) for i in range(blend_frames)], dim=0)
        alpha = torch.linspace(0, 1, blend_frames, device=prev_motion.device).view(-1, 1)
        blended = extrapolated * (1 - alpha) + curr_motion[:blend_frames] * alpha
        return torch.cat([prev_motion, blended, curr_motion[blend_frames:]], dim=0)

    @torch.no_grad()
    def sample_motion(
        self,
        windows,
        total_frames=None,
        window_size=64,
        guidance_scale=None,
        activity=None,
        init_past_motion=None,
        use_dead_blend=True,
        edit=None,
    ):
        """Generate a sequence window by window.

        windows           per-window conditions: {"rhythm", "semantic", "mel": [W, d], "clip_text": [512],
                          "shape_betas": [10], "trajectory_cond": [W, 4] (editable model)},
                          windows starting every window_size - overlap_size frames
        activity          [activity_dim] speaker id
        init_past_motion  [overlap, D] history for the first window (optional)
        edit              keypose editing of an existing sequence (editable model):
                          {"base", "target", "mask": [T, D] normalized wavelet motion / mask,
                           "start": noise level the repaint starts from (0..1), "strength": mask scale}.
                          Windows whose mask is empty keep the base motion.
        Returns the normalized wavelet motion [total_frames, D].
        """
        guidance_scale = self.hparams.guidance_scale if guidance_scale is None else guidance_scale
        overlap_size = self.hparams.overlap_size
        blend_frames = self.hparams.blend_frames
        stride = window_size - overlap_size
        past_motion = init_past_motion
        accumulated = None
        for window_idx, window in enumerate(windows):
            speech = {k: v.to(self.device).unsqueeze(0) for k, v in window.items() if k in SPEECH_KEYS}
            conditions = {k: v.to(self.device).unsqueeze(0) for k, v in window.items() if k not in SPEECH_KEYS}
            clean_past = None
            if past_motion is not None:
                clean_past = past_motion.to(self.device)
                clean_past = clean_past.unsqueeze(0) if clean_past.dim() == 2 else clean_past
            repaint = None
            if edit is not None:
                start = window_idx * stride
                base = self._window_slice(edit["base"], start, window_size).unsqueeze(0)
                mask = self._window_slice(edit["mask"], start, window_size).unsqueeze(0)
                mask = mask.clamp(0.0, 1.0) * float(edit["strength"])
                if clean_past is not None and overlap_size > 0:
                    mask[:, :overlap_size, :] = 0.0  # the clean history has priority
                if bool((mask.max() > 0).item()):
                    target = self._window_slice(edit["target"], start, window_size).unsqueeze(0)
                    repaint = {"base": base, "target": target, "mask": mask, "start": edit["start"]}
            if edit is not None and repaint is None:
                window_motion = base.squeeze(0)
            else:
                window_motion = self._sample_windows(
                    speech, conditions, activity.to(self.device).unsqueeze(0), clean_past,
                    guidance_scale, window_size, overlap_size, repaint=repaint,
                ).squeeze(0)
            if window_idx == 0:
                accumulated = window_motion.cpu()
            else:
                new_frames = window_motion[overlap_size:].cpu()
                if use_dead_blend and blend_frames > 0:
                    accumulated = self._dead_blend(accumulated, new_frames, blend_frames=blend_frames)
                else:
                    accumulated = torch.cat([accumulated, new_frames], dim=0)
            # the next window's history: the last committed frames
            past_motion = accumulated[-overlap_size:].clone()

        if total_frames is not None:
            if accumulated.shape[0] > total_frames:
                accumulated = accumulated[:total_frames]
            elif accumulated.shape[0] < total_frames:
                accumulated = torch.cat([accumulated, accumulated[-1:].repeat(total_frames - accumulated.shape[0], 1)], dim=0)
        return accumulated

    def _window_slice(self, seq, start, window_size):
        """seq[start:start + window_size] on the model device, padded by repeating its last frame."""
        seq = seq.to(self.device) if torch.is_tensor(seq) else torch.as_tensor(seq, device=self.device)
        segment = seq[start:start + window_size]
        if segment.shape[0] == 0:
            return torch.zeros(window_size, seq.shape[-1], device=self.device, dtype=seq.dtype)
        if segment.shape[0] < window_size:
            segment = torch.cat([segment, segment[-1:].repeat(window_size - segment.shape[0], 1)], dim=0)
        return segment
