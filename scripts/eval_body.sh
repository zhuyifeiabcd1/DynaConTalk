#!/bin/bash
# Evaluate a speech-only body checkpoint on the BEAT2 test split (EMAGE body metrics: FGD, BC, Diversity).
# Arguments are passed to src/tools/eval_body.py, e.g.
#   --checkpoint logs/<task>/runs/<time>/checkpoints/<name>.ckpt --beat2_root /path/to/BEAT2/beat_english_v2.0.0
set -e
cd "$(dirname "$0")/.."
: "${DYNACONTALK_DATA_DIR:?set DYNACONTALK_DATA_DIR to the preprocessed BEAT2 directory}"
export PROJECT_ROOT="$(pwd)"
python src/tools/eval_body.py "$@"
