#!/usr/bin/env python3
"""Darija ASR from a Hugging Face adapter repo.

The adapter repo holds LoRA (and MultiConv weights if hybrid). The 2B
Cohere base is pulled automatically. That base is **gated**: accept the
license on the model card, then log in. The Darija adapters themselves
are public.

    # one-time
    # 1. https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
    # 2. huggingface-cli login   OR   export HF_TOKEN=hf_...
    #    (huggingface-cli login writes ~/.cache/huggingface/token)

    python infer.py clip.wav
    python infer.py --clip clip.wav --model hybrid
    python infer.py clip.wav --model 01Yassine/cohere-transcribe-darija-full-lora

    from infer import transcribe
    print(transcribe("clip.wav"))
    print(transcribe("clip.wav", model_id="full_lora"))
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

BASE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
SAMPLE_RATE = 16000
LANGUAGE = "ar"

# Short names → public adapter repos. Each ships infer.py + adapters.py.
MODELS = {
    "hybrid": "01Yassine/cohere-transcribe-darija",
    "full_lora": "01Yassine/cohere-transcribe-darija-full-lora",
    "encoder_lora": "01Yassine/cohere-transcribe-darija-encoder-lora",
    "decoder_lora": "01Yassine/cohere-transcribe-darija-decoder-lora",
}
HUB_ID = MODELS["hybrid"]

_GATED_HINT = (
    "The Cohere base is gated. Accept the license at "
    "https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026 "
    "then `huggingface-cli login` or `export HF_TOKEN=hf_...`. "
    "The 01Yassine/darija adapters are public; the token is only for the 2B base."
)


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    for path in (
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
        Path("/netscratch/yelkheir/.cache/huggingface/token"),
        Path.home() / ".cache/huggingface/token",
    ):
        if path is not None and path.is_file():
            text = path.read_text().strip()
            if text:
                return text
    return None


def _hub_kw(**extra) -> dict:
    kw = dict(extra)
    token = _hf_token()
    if token:
        kw["token"] = token
    return kw


def _load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), int(sr), SAMPLE_RATE
        ).numpy()
    return np.asarray(wav, dtype=np.float32)


def resolve_id(model_id: str) -> str:
    return MODELS.get(model_id, model_id)


def _snapshot(model_id: str) -> Path:
    """Local dir, or download the Hub adapter repo (weights + adapters.py)."""
    local = Path(model_id)
    if local.is_dir() and (local / "adapter_config.json").exists():
        return local.resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(resolve_id(model_id), **_hub_kw()))
    except Exception as exc:
        raise SystemExit(f"{exc}\n\n{_GATED_HINT}") from exc


def _attach_conv(model, root: Path, meta: dict) -> None:
    if not meta.get("attached_layers") and not meta.get("conv_adapter"):
        return
    adapters_py = root / "adapters.py"
    if adapters_py.exists():
        sys.path.insert(0, str(root))
        attach = importlib.import_module("adapters").attach_multiconv_adapters
    else:
        from src.adapters import attach_multiconv_adapters as attach

    attach(
        model,
        bottleneck=meta["bottleneck"],
        kernels=tuple(meta["kernels"]),
        dropout=meta["dropout"],
        skip_bottom_frac=meta.get("skip_bottom_frac", 0.33),
        fusion=meta.get("fusion", "concat_fusion"),
        merge_kernel=meta.get("merge_kernel", 31),
    )


def load_model(model_id: str = HUB_ID, device: str | None = None):
    """Load Cohere Arabic + this Hub adapter. No local training files needed."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from peft import PeftModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = _snapshot(model_id)
    meta_path = root / "adapter_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL, **_hub_kw())
        model = CohereAsrForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, **_hub_kw()
        )
    except Exception as exc:
        msg = str(exc)
        if any(s in msg.lower() for s in ("401", "403", "gated", "authorized", "token")):
            raise SystemExit(f"{exc}\n\n{_GATED_HINT}") from exc
        raise
    _attach_conv(model, root, meta)
    if (root / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(root))
    extra = root / "encoder_adapters.pt"
    if extra.exists() and extra.stat().st_size > 2048:
        model.load_state_dict(
            torch.load(extra, map_location="cpu", weights_only=True), strict=False
        )
    model.to(device).eval()
    return model, processor, device


@torch.inference_mode()
def transcribe(
    audio: str | np.ndarray,
    model_id: str = HUB_ID,
    model=None,
    processor=None,
    device=None,
) -> str:
    if model is None or processor is None:
        model, processor, device = load_model(model_id, device)
    wav = _load_wav(audio) if isinstance(audio, str) else np.asarray(audio, dtype=np.float32)
    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=LANGUAGE)
    try:
        inputs = inputs.to(device, dtype=model.dtype)
    except (TypeError, AttributeError):
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dict(inputs).items()}
    out = model.generate(**inputs, max_new_tokens=128)
    chunk = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
    try:
        text = processor.decode(
            out, skip_special_tokens=True, audio_chunk_index=chunk, language=LANGUAGE
        )
        if isinstance(text, (list, tuple)):
            text = text[0]
    except TypeError:
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return str(text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Darija ASR from a Hugging Face adapter")
    parser.add_argument("audio", nargs="?", help="wav / flac / ogg path")
    parser.add_argument("--clip", default=None, help="same as the positional audio path")
    parser.add_argument(
        "--model",
        default="hybrid",
        help="Hub id, local dir, or one of: " + ", ".join(MODELS),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    clip = args.clip or args.audio
    if not clip:
        parser.error("pass a wav path, or --clip path.wav")
    if not _hf_token():
        print(
            "warning: no HF token found. The Darija adapters are public, but "
            f"{BASE_MODEL} is gated.\n{_GATED_HINT}",
            file=sys.stderr,
        )
    model, processor, device = load_model(args.model, args.device)
    print(transcribe(clip, model=model, processor=processor, device=device))


if __name__ == "__main__":
    main()
