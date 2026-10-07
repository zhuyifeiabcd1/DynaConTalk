#!/usr/bin/env python
"""Evaluate a DynaConTalk speech-only body model on the BEAT2 test split.

Reports the three body metrics of EMAGE (Liu et al., CVPR 2024), computed with the EMAGE
evaluation tools in src/models/emage_evaltools (FGD, BC, L1div):

  FGD        Frechet distance between AESKConv features of generated and ground-truth motion
  BC         beat consistency between motion beats and audio onsets
  Diversity  L1 diversity of joint positions (L1div)

Only speech-only body checkpoints are accepted (configs/dynacontalk_speech.yaml): the
editable model is also conditioned on keyposes and a root trajectory, which this
evaluation does not provide. Face models are evaluated with eval_face.py.

Every test sequence is generated end to end from speech: 64-frame windows with an 8-frame
overlap, no ground-truth frames, the sampler steps and guidance scale of the training config
(10 UniPC steps, 4.0). FGD is computed on 6D joint rotations; BC and Diversity on SMPL-X
joint positions (neutral body shape), BC leaving out the first and last 2 s of each sequence.
BC needs the BEAT2 audio (wave16k).

Usage:
  python src/tools/eval_body.py --checkpoint logs/<task>/runs/<time>/checkpoints/<name>.ckpt \
      --beat2_root /path/to/BEAT2/beat_english_v2.0.0
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

from src.models.emage_evaltools import BC, FGD, L1div  # noqa: E402
from src.models.emage_evaltools import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle  # noqa: E402
from src.models.emage_evaltools.motion_rep_transfer import get_motion_rep_numpy  # noqa: E402
from src.tools.eval_common import (  # noqa: E402
    find_config, identity_conditions, load_model, sequence_length, speech_windows,
)

JOINTS = 55
POSE_DIM = JOINTS * 6
LEVELS = 3
FPS = 30
BC_TRIM = 60  # frames left out at each end for BC (2 s)


def body_wavelet(wavelet, levels=LEVELS):
    """[T, 1732] interleaved coefficients -> body block [T, 1320]."""
    t = wavelet.shape[0]
    return wavelet.reshape(t, -1, levels + 1)[:, :POSE_DIM].reshape(t, -1)


def check_speech_only(checkpoint, model, cfg):
    if str(cfg.model.get("target_group", "body")) != "body":
        raise SystemExit(f"{checkpoint} is not a body model; evaluate face models with src/tools/eval_face.py")
    cond = model.denoiser.conditioning_module
    if getattr(cond, "use_trajectory_condition", False) or getattr(cond, "use_keypose_condition", False):
        raise SystemExit(
            f"{checkpoint} is an editable model (keypose / root-trajectory conditions). "
            "This evaluation generates from speech only and accepts speech-only checkpoints "
            "(configs/dynacontalk_speech.yaml)."
        )


@torch.no_grad()
def generate_poses(model, cfg, sample, data_dir, mean, std, device):
    """Generate one test sequence from speech; returns SMPL-X axis-angle poses [T, 165]."""
    window, overlap = int(cfg.data.window_size), int(cfg.data.overlap_size)
    total = sequence_length(sample)
    activity, shape = identity_conditions(sample, model, cfg, data_dir)
    windows = speech_windows(sample, total, window, overlap, shape)
    pred = model.sample_motion(windows, total_frames=total, window_size=window, activity=activity).to(device)
    length = min(pred.shape[0], total)
    pose6d = model._wavelet_to_motion((pred[:length] * std + mean).unsqueeze(0))[0, :, :POSE_DIM]
    return rotation_6d_to_axis_angle(pose6d.reshape(length, JOINTS, 6).float().cpu()).reshape(length, -1).numpy()


class BodyMetrics:
    """FGD / BC / L1div accumulated over sequences."""

    def __init__(self, device):
        evaltools = ROOT / "src" / "models" / "emage_evaltools"
        self.device = device
        self.fgd = FGD(download_path=str(evaltools), device=device)
        self.bc = BC(download_path=str(evaltools), sigma=0.3, order=7)
        self.l1div = L1div()

    def add(self, pred_poses, gt_poses, wav_path):
        t = min(pred_poses.shape[0], gt_poses.shape[0])
        pred_poses = np.asarray(pred_poses[:t], dtype=np.float32)
        gt_poses = np.asarray(gt_poses[:t], dtype=np.float32)
        position = get_motion_rep_numpy(pred_poses, device=self.device)["position"].reshape(t, -1)
        self.l1div.compute(position)
        if t > 2 * BC_TRIM:
            audio_beat = self.bc.load_audio(str(wav_path), t_start=int(BC_TRIM / FPS * 16000),
                                            t_end=int((t - BC_TRIM) / FPS * 16000))
            motion_beat = self.bc.load_motion(position, t_start=BC_TRIM, t_end=t - BC_TRIM, pose_fps=FPS,
                                              without_file=True)
            self.bc.compute(audio_beat, motion_beat, length=t - 2 * BC_TRIM, pose_fps=FPS)
        if t >= 32:
            pred6d = axis_angle_to_rotation_6d(torch.from_numpy(pred_poses).reshape(1, t, JOINTS, 3)).reshape(1, t, -1)
            gt6d = axis_angle_to_rotation_6d(torch.from_numpy(gt_poses).reshape(1, t, JOINTS, 3)).reshape(1, t, -1)
            self.fgd.update(pred6d.float(), gt6d.float())

    def results(self):
        return {"FGD": float(self.fgd.compute()), "BC": float(self.bc.avg()), "Diversity": float(self.l1div.avg())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="training config (default: config.yaml next to the checkpoint, or <run>/.hydra/config.yaml)")
    ap.add_argument("--data_dir", default=os.environ.get("DYNACONTALK_DATA_DIR"),
                    help="preprocessed BEAT2 directory (default: $DYNACONTALK_DATA_DIR)")
    ap.add_argument("--beat2_root", required=True, help="BEAT2 beat_english_v2.0.0 directory (for the audio)")
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
    wave_dir = Path(args.beat2_root) / "wave16k"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, cfg = load_model(checkpoint, config, data_dir, args.device)
    check_speech_only(checkpoint, model, cfg)
    mean = np.load(data_dir / "wavelet_mean.npy").astype(np.float32)
    std = np.load(data_dir / "wavelet_std.npy").astype(np.float32)
    std[std == 0] = 1e-6
    mean = torch.from_numpy(body_wavelet(mean[None])[0]).to(args.device)
    std = torch.from_numpy(body_wavelet(std[None])[0]).to(args.device)

    dataset = np.load(data_dir / "beat2_test.npy", allow_pickle=True).item()
    keys = sorted(dataset)[: args.limit or None]
    metrics = BodyMetrics(args.device)
    t0 = time.time()
    for i, key in enumerate(keys, 1):
        pred = generate_poses(model, cfg, dataset[key], data_dir, mean, std, args.device)
        metrics.add(pred, dataset[key]["poses"], wave_dir / f"{key}.wav")
        if i == 1 or i % 20 == 0 or i == len(keys):
            print(f"[{i}/{len(keys)}] {key}  {(time.time() - t0) / 60:.1f} min", flush=True)

    results = metrics.results()
    print(f"\nEMAGE body metrics, BEAT2 test ({len(keys)} sequences)")
    print(f"  FGD        {results['FGD']:.4f}")
    print(f"  BC         {results['BC']:.4f}")
    print(f"  Diversity  {results['Diversity']:.4f}")
    if args.out:
        record = {"checkpoint": str(checkpoint), "sequences": len(keys), "emage": results}
        Path(args.out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
