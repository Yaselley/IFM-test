#!/bin/bash
# Dedicated env for MOSS-Transcribe-Diarize and Cohere Transcribe.
# ifm-asr stays on transformers 4.57 (Whisper / qwen-asr). These two
# families need transformers>=5.4 (torch_compilable_check + CohereAsr).
set -euo pipefail

SRC="${IFM_ASR_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-asr}"
ENV_PATH="${IFM_MOSS_COHERE_ENV:-/netscratch/yelkheir/conda_ssl/envs/ifm-moss-cohere}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/netscratch/yelkheir/conda_ssl/pkgs}"

if [ ! -x "$SRC/bin/python" ]; then
  echo "Missing $SRC/bin/python" >&2
  exit 1
fi

if [ ! -x "$ENV_PATH/bin/python" ]; then
  echo "Creating venv at $ENV_PATH (reuses $SRC torch via system-site-packages)"
  rm -rf "$ENV_PATH"
  "$SRC/bin/python" -m venv --system-site-packages "$ENV_PATH"
else
  echo "Reusing $ENV_PATH"
fi

PY="$ENV_PATH/bin/python"
PIP="$ENV_PATH/bin/pip"
"$PY" -V
echo "prefix=$ENV_PATH"

"$PIP" install -U pip
# Do not install qwen-asr here — it pins transformers==4.57.6.
"$PIP" install -U \
  "transformers>=5.4,<6" \
  "accelerate>=0.33" \
  "huggingface_hub>=0.30" \
  sentencepiece \
  protobuf \
  soundfile \
  librosa \
  jiwer \
        pandas \
        numpy \
        tqdm \
        "peft>=0.14" \
        speechbrain

"$PY" - <<'PY'
import torch
import transformers
from transformers.utils import torch_compilable_check
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, transformers.__file__)
print("torch_compilable_check", torch_compilable_check)
print("CohereAsrForConditionalGeneration", CohereAsrForConditionalGeneration)
print("OK")
PY

echo
echo "Python: $PY"
echo "Score with:"
echo "  bash benchmarks/zeroshot_casablanca/run_moss_cohere.sh"
