#!/usr/bin/env python
"""Export a trained root-translation BiGRU for the Studio.

Reads a training checkpoint of configs/trajectory_bigru.yaml and writes the network weights
with their feature settings ({"state_dict", "config"}, readable with weights_only=True), the
file the Studio loads from checkpoints/trajectory_bigru/model.ckpt.

Usage:
  python src/tools/export_trajectory.py --checkpoint logs/trajectory_bigru/runs/<time>/checkpoints/last.ckpt \
      --out checkpoints/trajectory_bigru/model.ckpt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.trajectory_bigru import TrajectoryModule  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="training checkpoint (e.g. last.ckpt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module = TrajectoryModule(**ckpt["hyper_parameters"])
    module.load_state_dict(ckpt["state_dict"], strict=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.clone() for k, v in module.net.state_dict().items()}
    torch.save({"state_dict": state, "config": module.studio_config()}, out)
    print(f"{out}: epoch {ckpt.get('epoch')}, {sum(v.numel() for v in state.values())} parameters")


if __name__ == "__main__":
    main()
