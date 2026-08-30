#!/bin/bash
set -euo pipefail
ROOT="/netscratch/yelkheir/Personal/MBZUAI-IFM-test"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}:$ROOT/.deps"
export HF_HOME="${HF_HOME:-/netscratch/yelkheir/.cache/huggingface}"
export PYTHONNOUSERSITE=1
cd "$ROOT"
mkdir -p benchmarks/zeroshot_atlasia/results
exec python -u benchmarks/zeroshot_casablanca/run.py \
  --manifest data/splits/gold_atlasia.jsonl \
  --out-dir benchmarks/zeroshot_atlasia/results \
  "$@"
