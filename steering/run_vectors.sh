#!/usr/bin/env bash
# Compute Steer2Adapt's Big Five vectors with repeng and compare them to the published ones.
#   TRAIT=agreeableness GPU=1 bash steering/run_vectors.sh      (TRAIT=all for all five)
# Needs the `steer` env (repeng 0.4.0) and a clone of the Steer2Adapt repo at $S2A.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate steer
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TRAIT=${TRAIT:-agreeableness}
S2A=${S2A:-$HOME/NeurIPS-2026-Steer2Adapt}
# the meta-llama Hub cache is gone from the cluster; this is the same weights (unsloth's ungated copy)
MODEL_PATH=${MODEL_PATH:-$HOME/wassa_Luka/models/Llama-3.1-8B-Instruct}

python -m steering.compute_vectors --trait "$TRAIT" --model-path "$MODEL_PATH"
python -m steering.compare_vectors --theirs "$S2A/vectors" --trait "$TRAIT"
