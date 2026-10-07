#!/usr/bin/env python
"""Evaluate a DynaConTalk face model on the BEAT2 test split.

Reports the two face metrics of EMAGE (Liu et al., CVPR 2024), computed with the EMAGE
evaluation tools in src/models/emage_evaltools (MSEFace, LVDFace):

  MSE   mean squared error between predicted and ground-truth face vertices
  LVD   mean absolute difference between predicted and ground-truth vertex velocities

Only face checkpoints are accepted (configs/dynacontalk_face.yaml); body models are evaluated
with eval_body.py.

Vertices come from SMPL-X driven by the ground-truth jaw pose and body shape with the
predicted / ground-truth FLAME expressions (ground-truth expressions are reconstructed
from the stored wavelet coefficients). Every test sequence is generated end to end from
speech: 64-frame windows with an 8-frame overlap, the first 8 frames taken from the ground
truth, the sampler steps and guidance scale of the training config (10 UniPC steps, 4.0).

The model is built from the training config of the checkpoint (config.yaml next to it, or
<run>/.hydra/config.yaml for a checkpoint inside a training run) and its weights are
loaded strictly.

Usage:
  python src/tools/eval_face.py --checkpoint logs/<task>/runs/<time>/checkpoints/<name>.ckpt
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rootutils
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from src.models.emage_evaltools import LVDFace, MSEFace  # noqa: E402
from src.models.emage_evaltools.motion_rep_transfer import get_motion_rep_numpy  # noqa: E402
from src.models.wavelet import MotionWaveletISWT  # noqa: E402
from src.tools.eval_common import (  # noqa: E402
    find_config, identity_conditions, load_model, sequence_length, speech_windows,
)

POSE_DIM = 330  # 55 joints x 6D
TRANS_DIM = 3
FACE_DIM = 100
LEVELS = 3
NUM_BETAS = 300  # SMPL-X model shape space


def face_wavelet(wavelet, levels=LEVELS):
    """[T, 1732] interleaved coefficients -> face block [T, 400]."""
    t = wavelet.shape[0]
    return wavelet.reshape(t, -1, levels + 1)[:, POSE_DIM + TRANS_DIM:].reshape(t, -1)


def face_stats(data_dir):
    mean = face_wavelet(np.load(data_dir / "wavelet_mean.npy").astype(np.float32)[None])[0]
    std = face_wavelet(np.load(data_dir / "wavelet_std.npy").astype(np.float32)[None])[0]
    std[std == 0] = 1e-6
    return mean, std


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="training config (default: config.yaml next to the checkpoint, or <run>/.hydra/config.yaml)")
    ap.add_argument("--data_dir", default=os.environ.get("DYNACONTALK_DATA_DIR"),
                    help="preprocessed BEAT2 directory (default: $DYNACONTALK_DATA_DIR)")
    ap.add_argument("--out", default=None, help="write the metrics to this JSON file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="only the first N test sequences (for checking)")
    args = ap.parse_args()
    if not args.data_dir:
        raise SystemExit("set --data_dir or DYNACONTALK_DATA_DIR")

    checkpoint = Path(args.checkpoint)
    config = Path(args.config) if args.config else find_config(checkpoint)
    data_dir = Path(args.data_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, cfg = load_model(checkpoint, config, data_dir, args.device)
    if str(cfg.model.get("target_group", "body")) != "face":
        raise SystemExit(f"{checkpoint} is not a face model; evaluate body models with src/tools/eval_body.py")
    window, overlap = int(cfg.data.window_size), int(cfg.data.overlap_size)
    mean, std = face_stats(data_dir)
    iswt = MotionWaveletISWT(FACE_DIM, levels=LEVELS, wavelet="db6").to(args.device).eval()

    def to_expressions(face_wav):
        x = torch.from_numpy(face_wav.astype(np.float32)).T.unsqueeze(0).to(args.device)
        with torch.no_grad():
            return iswt(x).squeeze(0).T.cpu().numpy()

    dataset = np.load(data_dir / "beat2_test.npy", allow_pickle=True).item()
    keys = list(dataset)[: args.limit or None]
    mse, lvd = MSEFace(), LVDFace()
    t0 = time.time()
    for i, key in enumerate(keys, 1):
        s = dataset[key]
        total = sequence_length(s)
        gt_wav = face_wavelet(np.asarray(s["wavelet"][:total], dtype=np.float32))
        activity, shape = identity_conditions(s, model, cfg, data_dir)
        windows = speech_windows(s, total, window, overlap, shape)
        past = torch.from_numpy((gt_wav[:overlap] - mean) / (std + 1e-8)).to(args.device)
        with torch.no_grad():
            pred = model.sample_motion(windows, total_frames=total, window_size=window, activity=activity,
                                       init_past_motion=past).cpu().numpy()

        pred_expr = to_expressions(pred * std + mean)[:total]
        gt_expr = to_expressions(gt_wav)[:total]
        poses = np.asarray(s["poses"][:total], dtype=np.float32)
        betas = np.zeros(NUM_BETAS, dtype=np.float32)
        betas[: len(s["shape_betas"])] = s["shape_betas"]
        pv = get_motion_rep_numpy(poses, device=args.device, expressions=pred_expr, expression_only=True,
                                  betas=betas)["vertices"]
        gv = get_motion_rep_numpy(poses, device=args.device, expressions=gt_expr, expression_only=True,
                                  betas=betas)["vertices"]
        mse.compute(pv, gv)
        lvd.compute(pv, gv)
        if i == 1 or i % 20 == 0 or i == len(keys):
            print(f"[{i}/{len(keys)}] {key}  {(time.time() - t0) / 60:.1f} min", flush=True)

    results = {"MSE": float(mse.avg()), "LVD": float(lvd.avg())}
    print(f"\nEMAGE face metrics, BEAT2 test ({len(keys)} sequences)")
    print(f"  MSE  {results['MSE']:.3e}")
    print(f"  LVD  {results['LVD']:.3e}")
    if args.out:
        record = {"checkpoint": str(checkpoint), "sequences": len(keys), "emage": results}
        Path(args.out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
