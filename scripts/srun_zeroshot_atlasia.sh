#!/bin/bash
# AtlasIA zero-shot, two envs:
#   ifm-asr (transformers 4.57): Whisper, Qwen3-ASR, CTC
#   ifm-moss-cohere (transformers>=5.4): MOSS, Cohere
# Resume-safe: finished metrics.json with matching n is skipped.
set -euo pipefail
ROOT="/netscratch/yelkheir/Personal/MBZUAI-IFM-test"
ASR_ENV="${IFM_ASR_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-asr}"
COHERE_ENV="${IFM_MOSS_COHERE_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-moss-cohere}"
IMAGE="${IFM_CONTAINER:-/netscratch/yelkheir/containers/hg-ssl.sqsh}"
PARTITION="${1:-RTX3090,A100-PCI,A100-40GB,L40S}"
PHASE="${2:-}"   # asr | cohere | empty=both
HF_HOME_VAL="${HF_HOME:-/netscratch/yelkheir/.cache/huggingface}"
mkdir -p "$ROOT/benchmarks/zeroshot_atlasia/results"

srun_py() {
  local NAME="$1" PY="$2" LOG="$3"
  shift 3
  local SRUN=(srun
    --job-name="$NAME"
    --partition="$PARTITION"
    --gpus=1
    --cpus-per-task=8
    --mem=40G
    --time=01:00:00
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
    export PYTHONPATH='$ROOT:$ROOT/.deps'
    export HF_HOME='$HF_HOME_VAL'
    export PYTHONNOUSERSITE=1
    cd '$ROOT'
    $PY -u benchmarks/zeroshot_casablanca/run.py \
      --manifest data/splits/gold_atlasia.jsonl \
      --out-dir benchmarks/zeroshot_atlasia/results \
      $*
  "
}

if [[ -z "$PHASE" || "$PHASE" == "asr" ]]; then
  srun_py ifm-zs-atl-asr "$ASR_ENV/bin/python" \
    "$ROOT/benchmarks/zeroshot_atlasia/srun_asr.log" \
    --only whisper-tiny whisper-base whisper-small whisper-medium whisper-large-v2 \
           whisper-large-v3 whisper-large-v3-turbo \
           qwen3-asr-0.6b qwen3-asr-1.7b \
           wav2vec2-xlsr-darija mms-1b-all
fi
if [[ -z "$PHASE" || "$PHASE" == "cohere" ]]; then
  srun_py ifm-zs-atl-coh "$COHERE_ENV/bin/python" \
    "$ROOT/benchmarks/zeroshot_atlasia/srun_cohere.log" \
    --only moss-transcribe-0.9b cohere-transcribe cohere-transcribe-arabic
fi
