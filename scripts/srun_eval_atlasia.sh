#!/bin/bash
# Eval one Cohere checkpoint on atlasia/darija-asr-benchmark (114 clips).
set -euo pipefail
NAME="${1:?usage: srun_eval_atlasia.sh <name> [model_dir]}"
case "$NAME" in
  zeroshot) DEFAULT_MODEL="CohereLabs/cohere-transcribe-arabic-07-2026" ;;
  method) DEFAULT_MODEL="checkpoints/cohere-method/best" ;;
  full_lora) DEFAULT_MODEL="checkpoints/cohere-full-lora/best" ;;
  decoder_lora) DEFAULT_MODEL="checkpoints/cohere-decoder-lora/best" ;;
  encoder_lora) DEFAULT_MODEL="checkpoints/cohere-encoder-lora/best" ;;
  *) DEFAULT_MODEL="checkpoints/cohere-${NAME}/best" ;;
esac
MODEL="${2:-$DEFAULT_MODEL}"
PARTITION="${3:-RTX3090,A100-PCI,A100-40GB,V100-32GB,L40S}"
ROOT="/netscratch/yelkheir/Personal/MBZUAI-IFM-test"
ENV_PATH="${IFM_MOSS_COHERE_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-moss-cohere}"
IMAGE="${IFM_CONTAINER:-/netscratch/yelkheir/containers/hg-ssl.sqsh}"
OUT="$ROOT/checkpoints/atlasia_eval/${NAME}.json"
LOG="$ROOT/checkpoints/atlasia_eval/${NAME}.srun.log"
MANIFEST="$ROOT/data/splits/gold_atlasia.jsonl"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/netscratch/yelkheir/.cache/huggingface}"
export PYTHONNOUSERSITE=1
mkdir -p "$ROOT/checkpoints/atlasia_eval"

SRUN=(srun
  --job-name="ifm-atl-${NAME}"
  --partition="$PARTITION"
  --gpus=1
  --cpus-per-task=4
  --mem=32G
  --time=00:30:00
  --kill-on-bad-exit
  --output="$LOG"
)
if [[ -f "$IMAGE" ]]; then
  SRUN+=(
    --container-image="$IMAGE"
    --container-mounts=/ds:/ds,/ds-slt:/ds-slt,/netscratch/yelkheir:/netscratch/yelkheir
    --container-workdir="$ROOT"
  )
fi

"${SRUN[@]}" --export=ALL bash -lc "
  set -euo pipefail
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
  export PYTHONPATH='$ROOT' HF_HOME='${HF_HOME}' PYTHONNOUSERSITE=1
  PY='$ENV_PATH/bin/python'
  cd '$ROOT'
  \"\$PY\" -u scripts/eval_gold.py --model '$MODEL' --name '$NAME' --out '$OUT' --manifest '$MANIFEST'
"
