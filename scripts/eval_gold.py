#!/usr/bin/env python3
"""Score a Cohere student on a gold jsonl (path + text). Default: AtlasIA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import BOOTSTRAP_N, SAMPLE_RATE, SEED, SPLITS_DIR
from src.cohere_runtime import load_cohere_student, transcribe_cohere
from src.io_utils import load_mono_16k, write_json
from src.metrics import bootstrap_ci, compute_error_rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SPLITS_DIR / "gold_atlasia.jsonl",
    )
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"loading {args.model} on {args.device} | n={len(rows)} | {args.manifest}", flush=True)
    model, processor = load_cohere_student(args.model, args.device)

    refs, hyps, audio_s = [], [], 0.0
    t0 = time.perf_counter()
    for row in tqdm(rows, desc=args.name):
        wav = load_mono_16k(row["path"], SAMPLE_RATE)
        audio_s += float(wav.numel()) / SAMPLE_RATE
        hyps.append(transcribe_cohere(model, processor, wav, args.device))
        refs.append(row["text"])
    took = time.perf_counter() - t0

    raw = compute_error_rates(refs, hyps, normalized=False)
    norm = compute_error_rates(refs, hyps, normalized=True)
    cer_pt, cer_lo, cer_hi = bootstrap_ci(refs, hyps, metric="cer", n_boot=BOOTSTRAP_N, seed=SEED)
    wer_pt, wer_lo, wer_hi = bootstrap_ci(refs, hyps, metric="wer", n_boot=BOOTSTRAP_N, seed=SEED)

    payload = {
        "name": args.name,
        "model": args.model,
        "n": len(rows),
        "hours": audio_s / 3600.0,
        "cer_raw": raw.cer,
        "wer_raw": raw.wer,
        "cer_norm": norm.cer,
        "wer_norm": norm.wer,
        "cer_norm_ci95": [cer_lo, cer_pt, cer_hi],
        "wer_norm_ci95": [wer_lo, wer_pt, wer_hi],
        "rtf": took / max(audio_s, 1e-6),
        "latency_ms": 1000.0 * took / max(len(rows), 1),
        "hyps": [{"id": r["id"], "ref": a, "hyp": b} for r, a, b in zip(rows, refs, hyps)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {k: v for k, v in payload.items() if k != "hyps"})
    args.out.with_suffix(".hyps.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in payload["hyps"]) + "\n"
    )
    print(
        f"{args.name}: CER {norm.cer:.3f} [{cer_lo:.3f}, {cer_hi:.3f}]  "
        f"WER {norm.wer:.3f}  n={len(rows)} RTF {payload['rtf']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
