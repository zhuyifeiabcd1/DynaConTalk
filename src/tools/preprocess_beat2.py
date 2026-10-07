#!/usr/bin/env python
"""Build the DynaConTalk training dataset from raw BEAT2 (English).

Writes beat2_{train,val,test}.npy and the normalization statistics that the
released models were trained with.

Sequences: every speaker, BEAT2 official train/val/test split ("additional" excluded),
full length at 30 fps. Per sequence:

  poses, trans        SMPL-X axis-angle pose (165) and root translation (3)
  shape_betas         first 10 SMPL-X shape coefficients
  wavelet             3-level stationary wavelet transform (db6) of
                      [6D joint rotations (330), root translation relative to frame 0 (3),
                       FLAME expressions (100)], coefficients interleaved per channel -> 1732
  audio_features      semantic: HuBERT-Large (facebook/hubert-large-ls960-ft), 50 -> 30 fps
                      rhythm:   amplitude envelope, short-time energy, onsets (3)
                      mel:      128-band mel power spectrogram (no log)
  clip_text           CLIP ViT-B/32 embedding of the TextGrid transcript (512)
  clip_text_windows   the same per 64-frame window, stride 56
  activity_raw        [speaker id / 30, mean, variance of horizontal root speed]
  activity            activity_raw with the two speed terms z-scored on the training split

Arrays are stored as float16 (the format used for training; the data loader casts
back to float32). Pass --float32 to keep float32.

Usage:
  python src/tools/preprocess_beat2.py --beat2_root /path/to/BEAT2/beat_english_v2.0.0 \
      --out_dir /path/to/beat2_all_db6
"""
import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.speech_features import (  # noqa: E402
    encode_text, extract_hubert, extract_mel, extract_rhythm, load_clip, load_hubert,
)
from src.models.wavelet import MotionWaveletSWT  # noqa: E402

POSE_FPS = 30
WINDOW_SIZE = 64
WINDOW_STRIDE = 56
SHAPE_DIM = 10
WAVELET = "db6"
WAVELET_LEVELS = 3

F16_FIELDS = ("wavelet", "poses", "clip_text", "clip_text_windows")
F16_AUDIO = ("semantic", "mel", "rhythm")


# ---------------------------------------------------------------- audio features

def _fit_length(x, num_frames):
    if len(x) > num_frames:
        return x[:num_frames]
    if len(x) < num_frames:
        pad = np.zeros((num_frames - len(x), x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)
    return x


# ---------------------------------------------------------------- transcript / CLIP

def _normalize_word(word):
    if word is None:
        return ""
    w = str(word)
    for ch in ",.?!":
        w = w.replace(ch, " ")
    return " ".join(w.split()).strip()


def words_per_frame(textgrid_file, num_frames, pose_fps=POSE_FPS):
    import textgrid

    words = [""] * num_frames
    if not os.path.exists(textgrid_file):
        print(f"  [warn] TextGrid not found: {textgrid_file}")
        return words
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    tier = next((t for t in tg if t.name.lower() == "words"), tg[0])
    intervals = list(tier)
    for i in range(num_frames):
        t = i / pose_fps
        for interval in intervals:
            if interval.minTime <= t <= interval.maxTime:
                words[i] = _normalize_word(interval.mark)
                break
    return words


def sentence_of(words):
    """Drop blanks and consecutive duplicates, join with spaces."""
    dedup = []
    for w in (w for w in words if w is not None and str(w).strip()):
        if not dedup or dedup[-1] != w:
            dedup.append(w)
    return " ".join(dedup)


def window_starts(num_frames, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    if num_frames < window_size:
        return [0]
    return list(range(0, num_frames - window_size + 1, stride))


# ---------------------------------------------------------------- motion / wavelet

def _axis_angle_to_matrix(axis_angle):
    eps = 1e-8
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + eps)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype).view((1,) * len(axis_angle.shape[:-1]) + (3, 3)).expand_as(K)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + sin * K + (1 - cos) * (K @ K)


def axis_angle_to_rot6d(axis_angle):
    return _axis_angle_to_matrix(axis_angle)[..., :2, :].clone().reshape(*axis_angle.shape[:-1], 6)


def motion_wavelet(poses, trans, expressions, swt):
    T = poses.shape[0]
    rot6d = axis_angle_to_rot6d(torch.from_numpy(poses).float().view(T, 55, 3)).view(T, 330)
    trans_rel = torch.from_numpy(trans - trans[0:1]).float()
    expr = torch.from_numpy(expressions).float()
    motion = torch.cat([rot6d, trans_rel, expr], dim=-1).t().unsqueeze(0)  # [1, 433, T]
    coeff = swt(motion)  # [1, 433 * (L+1), T], band-major
    b, d, t = coeff.shape
    l1 = swt.levels + 1
    coeff = coeff.view(b, l1, d // l1, t).permute(0, 2, 1, 3).reshape(b, d, t)  # channel-major
    return coeff.squeeze(0).t().contiguous().numpy().astype(np.float32)


def activity_raw(video_id, trans):
    speaker_id = int(str(video_id).split("_")[0])
    root_xz = trans[:, [0, 2]]
    if root_xz.shape[0] > 1:
        speeds = np.linalg.norm(root_xz[1:] - root_xz[:-1], axis=-1) * POSE_FPS
        mean_speed = float(speeds.mean())
        var_speed = float(((speeds - mean_speed) ** 2).mean())
    else:
        mean_speed = var_speed = 0.0
    return np.array([speaker_id / 30.0, mean_speed, var_speed], dtype=np.float32)


# ---------------------------------------------------------------- one sequence

class Extractor:
    def __init__(self, beat2_root, device):
        # The released features were extracted in full fp32. TF32 (on by default for
        # cuDNN convolutions on Ampere and newer GPUs) shifts the HuBERT features by
        # about 2% of their standard deviation, so it is disabled here.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.root = Path(beat2_root)
        self.device = device
        self.hubert = load_hubert(device)
        self.clip_model, self.clip_module = load_clip(device)
        self.swt = MotionWaveletSWT(433, levels=WAVELET_LEVELS, wavelet=WAVELET)

    def build(self, key):
        npz = np.load(self.root / "smplxflame_30" / f"{key}.npz", allow_pickle=True)
        poses = np.asarray(npz["poses"], dtype=np.float32)
        trans = np.asarray(npz["trans"], dtype=np.float32)
        expressions = np.asarray(npz["expressions"], dtype=np.float32)
        betas = np.asarray(npz["betas"], dtype=np.float32).reshape(-1)[:SHAPE_DIM]
        num_frames = poses.shape[0]

        audio, _ = librosa.load(str(self.root / "wave16k" / f"{key}.wav"), sr=16000, mono=True)
        semantic = _fit_length(extract_hubert(*self.hubert, audio, self.device), num_frames)
        rhythm = _fit_length(extract_rhythm(audio), num_frames)
        mel = _fit_length(extract_mel(audio), num_frames)

        words = words_per_frame(str(self.root / "textgrid" / f"{key}.TextGrid"), num_frames)
        starts = window_starts(num_frames)
        clip_text = encode_text(self.clip_model, self.clip_module, sentence_of(words), self.device)
        clip_windows = np.stack([
            encode_text(self.clip_model, self.clip_module,
                        sentence_of(words[s:min(s + WINDOW_SIZE, num_frames)]), self.device)
            for s in starts
        ]).astype(np.float32)

        act = activity_raw(key, trans)
        return OrderedDict(
            trans=trans,
            activity_raw=act.copy(),
            clip_text_windows=clip_windows,
            activity=act,
            poses=poses,
            clip_text=clip_text,
            wavelet=motion_wavelet(poses, trans, expressions, self.swt),
            video_id=key,
            shape_betas=betas,
            window_size=WINDOW_SIZE,
            window_stride=WINDOW_STRIDE,
            window_starts=np.array(starts, dtype=np.int32),
            length=int(num_frames),
            audio_features={"rhythm": rhythm, "semantic": semantic, "mel": mel},
        )


def to_float16(sample):
    for f in F16_FIELDS:
        sample[f] = sample[f].astype(np.float16)
    for f in F16_AUDIO:
        sample["audio_features"][f] = sample["audio_features"][f].astype(np.float16)
    return sample


ORDER_FILE = Path(__file__).resolve().parents[2] / "data" / "beat2_sequence_order.json"


def read_split(beat2_root):
    """Official split, in the sequence order of the released training data.

    The data loader indexes training windows in dictionary order, so the order
    decides which windows a seeded shuffle groups into each batch.
    """
    split = {"train": [], "val": [], "test": []}
    with open(Path(beat2_root) / "train_test_split.csv") as f:
        for row in csv.DictReader(f):
            if row["type"] in split:
                split[row["type"]].append(row["id"])
    if ORDER_FILE.exists():
        import json

        order = json.loads(ORDER_FILE.read_text())
        for name, ids in split.items():
            if set(order[name]) != set(ids):
                raise RuntimeError(f"{ORDER_FILE.name}: {name} split does not match train_test_split.csv")
            split[name] = list(order[name])
    else:
        print(f"[warn] {ORDER_FILE} not found, using sorted ids; batches will differ from the released runs")
        split = {k: sorted(v) for k, v in split.items()}
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beat2_root", required=True, help="BEAT2 beat_english_v2.0.0 directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--float32", action="store_true", help="store float32 instead of float16")
    ap.add_argument("--keys", nargs="*", default=None, help="only process these sequence ids (for checking)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = read_split(args.beat2_root)
    if args.keys:
        wanted = set(args.keys)
        splits = {k: [x for x in v if x in wanted] for k, v in splits.items()}
    ext = Extractor(args.beat2_root, args.device)

    act_mean = act_std = None
    for split in ("train", "val", "test"):
        keys = splits[split]
        data = OrderedDict()
        wsum = wsq = None
        wcount = 0
        shapes = []
        t0 = time.time()
        for i, key in enumerate(keys):
            sample = ext.build(key)
            if split == "train":
                w = sample["wavelet"]
                if wsum is None:
                    wsum = np.zeros(w.shape[1], np.float64)
                    wsq = np.zeros(w.shape[1], np.float64)
                wsum += w.sum(axis=0, dtype=np.float64)
                wsq += (w ** 2).sum(axis=0, dtype=np.float64)
                wcount += w.shape[0]
                shapes.append(sample["shape_betas"])
            data[key] = sample if args.float32 else to_float16(sample)
            if (i + 1) % 20 == 0 or i + 1 == len(keys):
                print(f"[{split}] {i + 1}/{len(keys)}  {(time.time() - t0) / (i + 1):.1f}s/seq", flush=True)

        if split == "train" and keys:
            mean = wsum / wcount
            std = np.sqrt(np.maximum(wsq / wcount - mean ** 2, 1e-8))
            np.save(out / "wavelet_mean.npy", mean.astype(np.float32))
            np.save(out / "wavelet_std.npy", std.astype(np.float32))
            np.save(out / "wavelet_meta.npy", {"levels": WAVELET_LEVELS, "wavelet": WAVELET,
                                               "include_expressions": True, "channels": int(mean.shape[0]),
                                               "stats_split": "train"})
            shp = np.stack(shapes)
            s_mean = shp.mean(axis=0).astype(np.float32)
            s_std = shp.std(axis=0).astype(np.float32)
            s_std[s_std < 1e-6] = 1.0
            for name in ("shape_mean", "shape_betas_mean"):
                np.save(out / f"{name}.npy", s_mean)
            for name in ("shape_std", "shape_betas_std"):
                np.save(out / f"{name}.npy", s_std)
            raw = np.stack([s["activity_raw"] for s in data.values()]).astype(np.float64)
            act_mean = raw[:, 1:].mean(axis=0)
            act_std = np.maximum(raw[:, 1:].std(axis=0), 1e-6)
            np.save(out / "activity_stats.npy", {"mean_speed": act_mean[0], "std_speed": act_std[0],
                                                 "mean_variance": act_mean[1], "std_variance": act_std[1]})
        if act_mean is None:
            sys.exit("training split must be processed first (activity statistics)")
        for s in data.values():
            raw = s["activity_raw"].astype(np.float64)
            s["activity"] = np.array([raw[0], *((raw[1:] - act_mean) / act_std)], dtype=np.float32)

        np.save(out / f"beat2_{split}.npy", data, allow_pickle=True)
        print(f"[{split}] wrote {out / f'beat2_{split}.npy'} ({len(data)} sequences)", flush=True)
        del data


if __name__ == "__main__":
    main()
