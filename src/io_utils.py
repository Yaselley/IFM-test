"""JSONL manifests and audio load."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
import torchaudio


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_mono_16k(path: str, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


def hub_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    for path in (
        Path("/netscratch/yelkheir/.cache/huggingface/token"),
        Path.home() / ".cache/huggingface/token",
    ):
        if path.is_file():
            text = path.read_text().strip()
            if text:
                return text
    return None


def hub_kw(**extra) -> dict:
    kw = dict(extra)
    token = hub_token()
    if token:
        kw["token"] = token
    return kw
