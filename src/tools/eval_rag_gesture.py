#!/usr/bin/env python
"""Evaluate a DynaConTalk speech-only body model on the BEAT2 test split with the protocol of RAG-Gesture.

RAG-Gesture (Mughal et al., CVPR 2025) evaluates on all 25 BEAT2 speakers with every test
sequence cut into 10 s chunks. This script follows its evaluation script
(https://github.com/m-hamza-mughal/RAG-Gesture, tools/evaluate.py) and computes the metrics
with the EMAGE evaluation tools in src/models/emage_evaltools:

  FGD        Frechet distance between AESKConv features of generated and ground-truth chunks
             (called FID by RAG-Gesture)
  BeatAlign  EMAGE beat consistency with a velocity threshold of 0.3, leaving out 10 frames
             at each end of a chunk
  L1Div      L1 diversity of joint positions within each chunk
  Diversity  mean pairwise distance between the joint positions of all chunks, divided by the
             chunk length (RAG-Gesture tables show it x1000)

Chunks: from frame 0, non-overlapping 300 frames (10 s); the number of chunks comes from the
whole seconds of the shorter of audio and motion, and a shorter tail is dropped. Joint
positions come from SMPL-X with the neutral body shape and no translation (global orientation
included). The audio is read as RAG-Gesture reads it (librosa's default rate resampled to
16 kHz, each chunk stored as a 16-bit WAV and read back). The ground truth is the BEAT2 motion
subsampled to 15 fps and linearly interpolated back to 30 fps in 6D, as in RAG-Gesture's
pipeline; --raw_gt uses the original 30 fps motion instead.

Generation is the same as in eval_body.py: every test sequence is generated end to end from
speech at 30 fps without ground-truth frames, and cut into chunks afterwards. Only speech-only
body checkpoints are accepted.

Usage:
  python src/tools/eval_rag_gesture.py --checkpoint logs/<task>/runs/<time>/checkpoints/<name>.ckpt \
      --beat2_root /path/to/BEAT2/beat_english_v2.0.0
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import rootutils
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from src.models.emage_evaltools import BC, FGD, L1div  # noqa: E402
from src.models.emage_evaltools import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle  # noqa: E402
from src.models.emage_evaltools import motion_rep_transfer  # noqa: E402
from src.tools.eval_body import body_wavelet, check_speech_only, generate_poses  # noqa: E402
from src.tools.eval_common import find_config, load_model  # noqa: E402

JOINTS = 55
FPS = 30
CHUNK = 10 * FPS  # 10 s chunks
AUDIO_SR = 16000
AUDIO_CHUNK = 10 * AUDIO_SR
BEAT_TRIM = 10  # frames left out at each end of a chunk for BeatAlign
BEAT_THRESHOLD = 0.3


def round_trip(poses):
    """Axis-angle [T, 165] -> 6D [T, 330] and back, as RAG-Gesture's evaluation does."""
    rot6d = axis_angle_to_rotation_6d(torch.from_numpy(poses).float().reshape(-1, JOINTS, 3))
    return rot6d.reshape(-1, JOINTS * 6), rotation_6d_to_axis_angle(rot6d).reshape(-1, JOINTS * 3).numpy()


def rag_ground_truth(poses, chunk):
    """Ground-truth chunk as RAG-Gesture builds it: 15 fps frames, interpolated back to 30 fps in 6D."""
    start = chunk * CHUNK // 2
    p15 = torch.from_numpy(poses[::2][start:start + CHUNK // 2]).float()
    rot6d = axis_angle_to_rotation_6d(p15.reshape(-1, JOINTS, 3)).reshape(p15.shape[0], -1)
    rot6d = F.interpolate(rot6d.T.unsqueeze(0), scale_factor=2, mode="linear")[0].T
    return rotation_6d_to_axis_angle(rot6d.reshape(-1, JOINTS, 6)).reshape(-1, JOINTS * 3).numpy()


def load_audio(wav_path):
    """BEAT2 speech as RAG-Gesture's data loader reads it."""
    y, sr = librosa.load(str(wav_path))
    return librosa.resample(y, orig_sr=sr, target_sr=AUDIO_SR)


def chunk_audio(audio, chunk):
    """Audio of one chunk, stored as a 16-bit WAV and read back as RAG-Gesture's evaluation does."""
    buf = io.BytesIO()
    sf.write(buf, audio[chunk * AUDIO_CHUNK:(chunk + 1) * AUDIO_CHUNK], AUDIO_SR, format="WAV")
    buf.seek(0)
    y, sr = librosa.load(buf)
    return librosa.resample(y, orig_sr=sr, target_sr=AUDIO_SR)[:AUDIO_CHUNK]


def num_chunks(audio, poses, expressions):
    seconds = min(audio.shape[0] // AUDIO_SR, poses[::2].shape[0] // 15, expressions[::2].shape[0] // 15)
    return seconds // 10


class RagGestureMetrics:
    """FGD / BeatAlign / L1Div / Diversity accumulated over 10 s chunks."""

    def __init__(self, device, raw_gt=False):
        evaltools = ROOT / "src" / "models" / "emage_evaltools"
        self.device = device
        self.raw_gt = raw_gt
        self.fgd = FGD(download_path=str(evaltools), device=device)
        self.beat = {"pred": BC(download_path=str(evaltools), sigma=0.3, order=7),
                     "gt": BC(download_path=str(evaltools), sigma=0.3, order=7)}
        for bc in self.beat.values():
            bc.threshold = BEAT_THRESHOLD
        self.l1div = {"pred": L1div(), "gt": L1div()}
        self.positions = {"pred": [], "gt": []}
        self.skipped = []

    @torch.no_grad()
    def joint_positions(self, poses):
        """SMPL-X joints [T, 55 * 3]: neutral shape, no translation, global orientation included."""
        model = motion_rep_transfer.smplx_model.to(self.device)
        p = torch.from_numpy(poses).float().to(self.device)
        n = p.shape[0]
        out = model(betas=torch.zeros(n, 300, device=self.device), transl=torch.zeros(n, 3, device=self.device),
                    expression=torch.zeros(n, 100, device=self.device), global_orient=p[:, :3],
                    body_pose=p[:, 3:66], jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
                    left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165])
        return out["joints"][:, :JOINTS].reshape(n, -1).cpu().numpy()

    def add_sequence(self, beat2_root, key, pred_poses):
        raw = np.load(Path(beat2_root) / "smplxflame_30" / f"{key}.npz")
        gt_poses = raw["poses"].astype(np.float32)
        audio = load_audio(Path(beat2_root) / "wave16k" / f"{key}.wav")
        for c in range(num_chunks(audio, gt_poses, raw["expressions"])):
            start, end = c * CHUNK, (c + 1) * CHUNK
            if end > pred_poses.shape[0]:
                self.skipped.append(f"{key}/{c}")
                continue
            gt = gt_poses[start:end] if self.raw_gt else rag_ground_truth(gt_poses, c)
            self.add_chunk(pred_poses[start:end], gt, chunk_audio(audio, c))

    def add_chunk(self, pred_poses, gt_poses, audio):
        n = pred_poses.shape[0]
        pred6d, pred_poses = round_trip(pred_poses)
        gt6d, gt_poses = round_trip(gt_poses)
        self.fgd.update(pred6d.unsqueeze(0).to(self.device), gt6d.unsqueeze(0).to(self.device))

        trim = int(BEAT_TRIM * AUDIO_SR / FPS)
        onsets = self.beat["pred"].load_audio(audio, t_start=trim, t_end=len(audio) - trim, without_file=True)
        for name, poses in (("pred", pred_poses), ("gt", gt_poses)):
            position = self.joint_positions(poses)
            self.l1div[name].compute(position.copy())
            self.positions[name].append(position)
            beats = self.beat[name].load_motion(position, t_start=BEAT_TRIM, t_end=n - BEAT_TRIM, pose_fps=FPS,
                                                without_file=True)
            self.beat[name].compute(onsets, beats, length=n - 2 * BEAT_TRIM, pose_fps=FPS)

    def diversity(self, name):
        x = torch.from_numpy(np.stack(self.positions[name]).reshape(len(self.positions[name]), -1))
        x = x.double().to(self.device)
        n, total = x.shape[0], 0.0
        for s in range(0, n, 256):
            total += float(torch.cdist(x[s:s + 256], x).sum())
        return total / 2 / CHUNK / (n * (n - 1) / 2)

    def results(self):
        pred = torch.cat(self.fgd.pred_features).numpy()
        gt = torch.cat(self.fgd.target_features).numpy()
        return {
            "chunks": len(self.positions["pred"]),
            "FGD": float(FGD.frechet_distance(pred, gt, eps=0.0)),
            "BeatAlign": float(self.beat["pred"].avg()),
            "L1Div": float(self.l1div["pred"].avg()),
            "Diversity": self.diversity("pred"),
            "ground_truth": {"BeatAlign": float(self.beat["gt"].avg()), "L1Div": float(self.l1div["gt"].avg()),
                             "Diversity": self.diversity("gt")},
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="training config (default: config.yaml next to the checkpoint, or <run>/.hydra/config.yaml)")
    ap.add_argument("--data_dir", default=os.environ.get("DYNACONTALK_DATA_DIR"),
                    help="preprocessed BEAT2 directory (default: $DYNACONTALK_DATA_DIR)")
    ap.add_argument("--beat2_root", required=True, help="BEAT2 beat_english_v2.0.0 directory (motion and audio)")
    ap.add_argument("--raw_gt", action="store_true",
                    help="compare with the original 30 fps ground truth instead of RAG-Gesture's 15 fps resampling")
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
    check_speech_only(checkpoint, model, cfg)
    mean = np.load(data_dir / "wavelet_mean.npy").astype(np.float32)
    std = np.load(data_dir / "wavelet_std.npy").astype(np.float32)
    std[std == 0] = 1e-6
    mean = torch.from_numpy(body_wavelet(mean[None])[0]).to(args.device)
    std = torch.from_numpy(body_wavelet(std[None])[0]).to(args.device)

    dataset = np.load(data_dir / "beat2_test.npy", allow_pickle=True).item()
    keys = sorted(dataset)[: args.limit or None]
    metrics = RagGestureMetrics(args.device, raw_gt=args.raw_gt)
    t0 = time.time()
    for i, key in enumerate(keys, 1):
        pred = generate_poses(model, cfg, dataset[key], data_dir, mean, std, args.device)
        metrics.add_sequence(args.beat2_root, key, pred)
        if i == 1 or i % 20 == 0 or i == len(keys):
            print(f"[{i}/{len(keys)}] {key}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if metrics.skipped:
        print(f"warning: {len(metrics.skipped)} chunks longer than the generated motion were left out")

    results = metrics.results()
    gt = results["ground_truth"]
    print(f"\nRAG-Gesture body metrics, BEAT2 test ({len(keys)} sequences, {results['chunks']} chunks of 10 s, "
          f"{'original 30 fps' if args.raw_gt else 'RAG-Gesture 15 fps'} ground truth)")
    print(f"  FGD        {results['FGD']:.4f}")
    print(f"  BeatAlign  {results['BeatAlign']:.4f}   (ground truth {gt['BeatAlign']:.4f})")
    print(f"  L1Div      {results['L1Div']:.4f}   (ground truth {gt['L1Div']:.4f})")
    print(f"  Diversity  {results['Diversity'] * 1e3:.2f}   (ground truth {gt['Diversity'] * 1e3:.2f}; x1000, as RAG-Gesture tabulates it)")
    if args.out:
        record = {"checkpoint": str(checkpoint), "sequences": len(keys), "raw_gt": args.raw_gt, "rag_gesture": results}
        Path(args.out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
