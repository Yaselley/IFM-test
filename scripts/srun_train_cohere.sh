#!/bin/bash
# LoRA decoder + MultiConvAdapter encoder on Cohere Transcribe Arabic.
# Recipe is sized for one 16GB card. Use ifm-moss-cohere (transformers>=5.4).
set -euo pipefail
if [[ "${1:-}" =~ ^(method|decoder_lora|full_lora|encoder_lora)$ ]]; then
  RECIPE="$1"
  PARTITION="${2:-A100-PCI,A100-40GB,A100-80GB,L40S,H100-PCI,RTX3090}"
else
  RECIPE="method"
  PARTITION="${1:-A100-PCI,A100-40GB,A100-80GB,L40S,H100-PCI,RTX3090}"
fi
ROOT="/netscratch/yelkheir/Personal/MBZUAI-IFM-test"
ENV_PATH="${IFM_MOSS_COHERE_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-moss-cohere}"
IMAGE="${IFM_CONTAINER:-/netscratch/yelkheir/containers/hg-ssl.sqsh}"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/netscratch/yelkheir/.cache/huggingface}"
export PYTHONNOUSERSITE=1
if [[ -z "${HF_TOKEN:-}" && -f /netscratch/yelkheir/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /netscratch/yelkheir/.cache/huggingface/token)"
fi

SRUN=(srun
  --job-name="ifm-cohere-${RECIPE}"
  --partition="$PARTITION"
  --gpus=1
  --cpus-per-task=8
  --mem=40G
  --time=03:00:00
  --kill-on-bad-exit
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
  export PYTHONPATH='$ROOT'
  export HF_HOME='${HF_HOME}'
  export PYTHONNOUSERSITE=1
  if [ -z \"\${HF_TOKEN:-}\" ] && [ -f /netscratch/yelkheir/.cache/huggingface/token ]; then
    export HF_TOKEN=\"\$(cat /netscratch/yelkheir/.cache/huggingface/token)\"
  fi
  PY='$ENV_PATH/bin/python'
  if [ ! -x \"\$PY\" ]; then
    echo \"Missing $ENV_PATH. Run scripts/setup_moss_cohere_env.sh first.\" >&2
    exit 1
  fi
  cd '$ROOT'
  \"\$PY\" -u -m src.train_cohere --recipe '$RECIPE'
"
