#!/bin/bash
# Evaluate a face checkpoint on the BEAT2 test split (EMAGE face metrics: MSE, LVD).
# Arguments are passed to src/tools/eval_face.py, e.g.
#   --checkpoint logs/<task>/runs/<time>/checkpoints/<name>.ckpt --out face_metrics.json
set -e
cd "$(dirname "$0")/.."
: "${DYNACONTALK_DATA_DIR:?set DYNACONTALK_DATA_DIR to the preprocessed BEAT2 directory}"
export PROJECT_ROOT="$(pwd)"
python src/tools/eval_face.py "$@"
