#!/usr/bin/env python3
"""Cache atlasia/darija-asr-benchmark wavs + jsonl on netscratch."""

from __future__ import annotations

import io
import json
from pathlib import Path

import soundfile as sf
import torchaudio
from datasets import Audio, load_dataset

from src.config import DATA_DIR, SAMPLE_RATE, SPLITS_DIR

OUT_DIR = DATA_DIR / "cache" / "atlasia"
JSONL = SPLITS_DIR / "gold_atlasia.jsonl"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("atlasia/darija-asr-benchmark", split="train").cast_column(
        "audio", Audio(decode=False)
    )
    rows = []
    for i, ex in enumerate(ds):
        src = io.BytesIO(ex["audio"]["bytes"]) if ex["audio"].get("bytes") else ex["audio"]["path"]
        wav, sr = sf.read(src, dtype="float32")
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        tensor = torchaudio.functional.resample(
            __import__("torch").tensor(wav).unsqueeze(0), int(sr), SAMPLE_RATE
        ) if int(sr) != SAMPLE_RATE else __import__("torch").tensor(wav).unsqueeze(0)
        path = OUT_DIR / f"{i:04d}.wav"
        torchaudio.save(str(path), tensor, SAMPLE_RATE)
        text = str(ex.get("text") or "").strip()
        duration = float(tensor.shape[-1]) / SAMPLE_RATE
        rows.append(
            {
                "id": f"atlasia_{i:04d}",
                "path": str(path),
                "text": text,
                "duration": duration,
                "split": "gold_test",
                "source": "atlasia/darija-asr-benchmark",
            }
        )
    JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600.0
    print(f"wrote {JSONL} n={len(rows)} hours={hours:.3f} dir={OUT_DIR}")


if __name__ == "__main__":
    main()
