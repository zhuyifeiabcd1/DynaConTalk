"""Shared helpers of the evaluation scripts (eval_body.py, eval_face.py)."""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def find_config(checkpoint):
    """Training config of a checkpoint: config.yaml next to it, or <run>/.hydra/config.yaml."""
    ckpt = Path(checkpoint).resolve()
    for cand in (ckpt.parent / "config.yaml", ckpt.parent.parent / ".hydra" / "config.yaml"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no training config found for {ckpt}; pass --config")


def load_model(checkpoint, config, data_dir, device):
    """Build the model from its training config and load the weights strictly."""
    import hydra

    cfg = OmegaConf.load(config)
    cfg.data.data_dir = str(data_dir)
    cfg.paths.output_dir = str(Path(checkpoint).resolve().parent)
    model = hydra.utils.instantiate(cfg.model)
    state = dict(torch.load(str(checkpoint), map_location="cpu", weights_only=False)["state_dict"])
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint({"state_dict": state})
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the config: missing={len(missing)} unexpected={len(unexpected)}")
    # Frozen parameters, as for the EMA network used in validation; this also lets PyTorch
    # use its fused Transformer inference kernels, matching the released evaluation exactly.
    model = model.to(device).eval().requires_grad_(False)
    model.sample_scheduler.set_timesteps(int(model.hparams.step_num))
    return model, cfg


def sequence_length(sample):
    return int(min(sample["length"], len(sample["poses"]), len(sample["wavelet"]),
                   *(v.shape[0] for v in sample["audio_features"].values())))


def speech_windows(sample, total, window, overlap, shape=None):
    """Per-window conditioning: speech features, CLIP text and (optionally) normalized body shape."""
    stride = window - overlap
    n = 1 if total <= window else 1 + (total - window + stride - 1) // stride
    starts_text = np.asarray(sample["window_starts"]).reshape(-1)
    text_windows = np.asarray(sample["clip_text_windows"], dtype=np.float32)
    windows = []
    for i in range(n):
        start = i * stride
        end = min(start + window, total)
        w = {}
        for key, val in sample["audio_features"].items():
            seg = torch.from_numpy(np.asarray(val[start:end], dtype=np.float32))
            if seg.shape[0] < window:  # pad the last window by repeating its final frame
                seg = torch.cat([seg, seg[-1:].repeat(window - seg.shape[0], *[1] * (seg.dim() - 1))])
            w[key] = seg
        nearest = int(np.argmin(np.abs(starts_text[: len(text_windows)] - start)))
        w["clip_text"] = torch.from_numpy(text_windows[nearest])
        if shape is not None:
            w["shape_betas"] = shape
        windows.append(w)
    return windows


def identity_conditions(sample, model, cfg, data_dir):
    """Speaker (activity) and normalized body shape, as the training data loader provides them."""
    activity_dim = int(cfg.data.activity_dim)
    activity = torch.from_numpy(np.asarray(sample["activity"], dtype=np.float32).reshape(-1)[:activity_dim])
    mean = np.load(Path(data_dir) / "shape_betas_mean.npy").astype(np.float32)
    std = np.load(Path(data_dir) / "shape_betas_std.npy").astype(np.float32)
    shape = torch.from_numpy((np.asarray(sample["shape_betas"], dtype=np.float32) - mean) / (std + 1e-8))
    return activity, shape
