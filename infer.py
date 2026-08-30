#!/usr/bin/env python3
"""Transcribe Darija audio with the adapted Cohere checkpoint.

    python infer.py clip.wav
    python infer.py clip.wav --model checkpoints/cohere-method/best
    python infer.py clip.wav --model 01Yassine/cohere-transcribe-darija
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HUB_ID = "01Yassine/cohere-transcribe-darija"
BASE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
SAMPLE_RATE = 16000
LANGUAGE = "ar"


def _load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), int(sr), SAMPLE_RATE
        ).numpy()
    return np.asarray(wav, dtype=np.float32)


def _resolve(model_id: str) -> Path:
    local = Path(model_id)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id))


def _attach(model, root: Path):
    meta = json.loads((root / "adapter_meta.json").read_text())
    try:
        from src.adapters import attach_multiconv_adapters
    except ImportError:
        sys.path.insert(0, str(root))
        from adapters import attach_multiconv_adapters  # type: ignore

    if meta.get("attached_layers"):
        attach_multiconv_adapters(
            model,
            bottleneck=meta["bottleneck"],
            kernels=tuple(meta["kernels"]),
            dropout=meta["dropout"],
            skip_bottom_frac=meta.get("skip_bottom_frac", 0.33),
            fusion=meta.get("fusion", "concat_fusion"),
            merge_kernel=meta.get("merge_kernel", 31),
        )
    return meta


def load_model(model_id: str = HUB_ID, device: str | None = None):
    """Load base Cohere + conv adapters + decoder LoRA."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from peft import PeftModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = _resolve(model_id)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16
    )
    _attach(model, root)
    if (root / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(root))
    extra = root / "encoder_adapters.pt"
    if extra.exists():
        model.load_state_dict(torch.load(extra, map_location="cpu", weights_only=True), strict=False)
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
        text = processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunk, language=LANGUAGE)
        if isinstance(text, (list, tuple)):
            text = text[0]
    except TypeError:
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return str(text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Darija ASR (adapted Cohere Transcribe Arabic)")
    parser.add_argument("audio", help="wav / flac / ogg path")
    parser.add_argument("--model", default=HUB_ID)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model, processor, device = load_model(args.model, args.device)
    print(transcribe(args.audio, model=model, processor=processor, device=device))


if __name__ == "__main__":
    main()
