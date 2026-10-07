#!/bin/bash
# Train the DynaConTalk face model with the released speech-only recipe.
# Extra arguments are passed to Hydra, e.g.
#   dgn=v1
#   trainer.devices=2 trainer.strategy=ddp_find_unused_parameters_true data.batch_size=96 trainer.accumulate_grad_batches=1
set -e
cd "$(dirname "$0")/.."
: "${DYNACONTALK_DATA_DIR:?set DYNACONTALK_DATA_DIR to the preprocessed BEAT2 directory}"
export PROJECT_ROOT="$(pwd)"
python src/train.py -cn dynacontalk_face "$@"
