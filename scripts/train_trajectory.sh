#!/bin/bash
# Train the root-translation BiGRU of the Studio on the official BEAT2 train split.
# Build its data first:
#   python src/tools/preprocess_trajectory.py --beat2_root /path/to/BEAT2/beat_english_v2.0.0 --out_dir $DYNACONTALK_TRAJECTORY_DIR
# Extra arguments are passed to Hydra.
set -e
cd "$(dirname "$0")/.."
: "${DYNACONTALK_TRAJECTORY_DIR:?set DYNACONTALK_TRAJECTORY_DIR to the output of src/tools/preprocess_trajectory.py}"
export PROJECT_ROOT="$(pwd)"
python src/train.py -cn trajectory_bigru "$@"
