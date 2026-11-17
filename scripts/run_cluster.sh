#!/usr/bin/env bash
# MICM-HPC-01 launcher (no SLURM): `steer` conda env, one GPU, offline HF cache.
# Code lives in ~/wassa-engagement (NOT ~/engagement, which is the older NoXI project).
#   setsid nohup scripts/run_cluster.sh > logs/baseline.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/miniconda3/envs/steer/bin:$PATH
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CFG=${1:-configs/baseline.yaml}
python -m engagement prepare  --config "$CFG"
python -m engagement features --config "$CFG"
for head in tfidf_ridge ridge linear mlp; do
  python -m engagement train --config "$CFG" --head "$head"
done
# what keeping the 14 test articles in train does (same val articles; CV part would be identical)
for head in tfidf_ridge ridge linear mlp; do
  python -m engagement train --config "$CFG" --head "$head" \
    --set data.exclude_test_articles_from_train=false --set "eval.protocols=[holdout]"
done
echo ALL_DONE
