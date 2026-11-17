#!/usr/bin/env bash
# Full baseline on a GPU node: data -> frozen-LLaMA features -> every head.
# Needs HF_TOKEN with access to meta-llama/Llama-3.1-8B-Instruct (gated).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/baseline.yaml}

[ -f data/raw/track3_train.csv ] || bash scripts/download_data.sh
python -m engagement prepare  --config "$CFG"
python -m engagement features --config "$CFG"          # ~100 forward passes, cached
for head in tfidf_ridge ridge linear mlp; do
  python -m engagement train --config "$CFG" --head "$head"
done
