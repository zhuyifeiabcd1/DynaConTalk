#!/usr/bin/env python
"""Build the training data of the root-translation BiGRU from raw BEAT2.

For every sequence of the official train and validation splits (BEAT2 train_test_split.csv),
computes the trajectory features from the SMPL-X pose (src/data/trajectory_dataset.py) and
stores them with the root translation:

  <out_dir>/trajectory_train.npy   official train split
  <out_dir>/trajectory_val.npy     official validation split

Usage:
  python src/tools/preprocess_trajectory.py --beat2_root /path/to/BEAT2/beat_english_v2.0.0 \
      --out_dir /path/to/trajectory_data
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.trajectory_dataset import trajectory_features  # noqa: E402

MIN_LENGTH = 32  # frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beat2_root", required=True, help="BEAT2 beat_english_v2.0.0 directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,val", help="official splits to build (comma separated)")
    ap.add_argument("--keys", default=None, help="only these sequences (comma separated; for checking)")
    args = ap.parse_args()

    beat2 = Path(args.beat2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(beat2 / "train_test_split.csv") as f:
        split_of = {row["id"]: row["type"] for row in csv.DictReader(f)}
    only = set(args.keys.split(",")) if args.keys else None

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        keys = sorted(k for k, s in split_of.items() if s == split and (only is None or k in only))
        out = {}
        for key in tqdm(keys, desc=split):
            motion = np.load(beat2 / "smplxflame_30" / f"{key}.npz")
            poses = motion["poses"].astype(np.float32)
            trans = motion["trans"].astype(np.float32)
            feat = trajectory_features(poses)
            length = min(feat.shape[0], trans.shape[0])
            if length < MIN_LENGTH:
                continue
            out[key] = {"input_feat": feat[:length].astype(np.float32), "trans": trans[:length]}
        path = out_dir / f"trajectory_{split}.npy"
        np.save(path, out, allow_pickle=True)
        print(f"{path}: {len(out)} sequences, {sum(v['trans'].shape[0] for v in out.values())} frames")


if __name__ == "__main__":
    main()
