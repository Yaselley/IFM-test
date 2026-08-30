"""Eval on silver / Casablanca. CER first, bootstrap CIs, slices."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    BOOTSTRAP_N,
    RESULTS_DIR,
    SAMPLE_RATE,
    SEED,
    SPLITS_DIR,
    WHISPER_LANGUAGE,
    WHISPER_TASK,
    WORST_N,
    ensure_dirs,
)
from src.io_utils import load_mono_16k, read_jsonl, write_json
from src.metrics import (
    bootstrap_ci,
    compute_error_rates,
    paired_bootstrap_delta,
    per_utt_cer,
    per_utt_wer,
)
from src.normalize import has_latin_codeswitch, normalize_text


def transcribe_whisper(model, processor, wav, device: str) -> str:
    inputs = processor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        predicted = model.generate(
            inputs.input_features.to(device),
            language=WHISPER_LANGUAGE,
            task=WHISPER_TASK,
            do_sample=False,
            num_beams=1,
        )
    return processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()


def transcribe_ctc(model, processor, wav, device: str) -> str:
    inputs = processor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()


def transcribe_qwen(model, wav, path: str | None = None) -> str:
    result = model.transcribe(audio=path if path else wav.numpy(), language="Arabic")
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def load_asr(model_path: str, kind: str, device: str):
    if kind == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_path)
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            from peft import PeftModel
            import json as _json

            cfg = _json.loads(adapter_cfg.read_text())
            base_id = cfg.get("base_model_name_or_path") or "openai/whisper-small"
            base = WhisperForConditionalGeneration.from_pretrained(base_id)
            model = PeftModel.from_pretrained(base, model_path)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return "whisper", model, processor
    if kind == "ctc":
        from transformers import AutoModelForCTC, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForCTC.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return "ctc", model, processor
    if kind == "qwen3_asr":
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(model_path, dtype="bfloat16", device_map=device)
        return "qwen3_asr", model, None
    if kind == "cohere":
        from src.cohere_runtime import load_cohere_student

        model, processor = load_cohere_student(model_path, device)
        return "cohere", model, processor
    raise ValueError(kind)


def predict_one(kind, model, processor, path: str, device: str) -> str:
    wav = load_mono_16k(path, SAMPLE_RATE)
    if kind == "whisper":
        return transcribe_whisper(model, processor, wav, device)
    if kind == "ctc":
        return transcribe_ctc(model, processor, wav, device)
    if kind == "cohere":
        from src.cohere_runtime import transcribe_cohere

        return transcribe_cohere(model, processor, wav, device)
    return transcribe_qwen(model, wav, path=path)


def slice_table(frame: pd.DataFrame) -> dict:
    out = {}

    def _pack(name: str, sub: pd.DataFrame) -> None:
        if len(sub) < 8:
            return
        raw = compute_error_rates(sub["text"], sub["hyp"], normalized=False)
        norm = compute_error_rates(sub["text"], sub["hyp"], normalized=True)
        out[name] = {
            "n": int(len(sub)),
            "hours": float(sub["duration"].sum() / 3600.0),
            "wer_raw": raw.wer,
            "cer_raw": raw.cer,
            "wer_norm": norm.wer,
            "cer_norm": norm.cer,
        }

    _pack("all", frame)
    if "has_cs" in frame:
        _pack("codeswitch", frame[frame["has_cs"] == True])  # noqa: E712
        _pack("no_codeswitch", frame[frame["has_cs"] == False])  # noqa: E712
    if frame["pesq_hyp"].notna().any():
        _pack("pesq_2.5_3.0", frame[frame["pesq_hyp"] < 3.0])
        _pack("pesq_3.0_3.5", frame[(frame["pesq_hyp"] >= 3.0) & (frame["pesq_hyp"] < 3.5)])
        _pack("pesq_3.5_plus", frame[frame["pesq_hyp"] >= 3.5])
    _pack("dur_3_6s", frame[frame["duration"] < 6])
    _pack("dur_6_10s", frame[(frame["duration"] >= 6) & (frame["duration"] < 10)])
    _pack("dur_10_15s", frame[frame["duration"] >= 10])
    if "num_speakers" in frame and frame["num_speakers"].notna().any():
        _pack("spk_1", frame[frame["num_speakers"] == 1])
        _pack("spk_ge2", frame[frame["num_speakers"] >= 2])
    if "channel" in frame and frame["channel"].nunique() <= 30:
        for ch, sub in frame.groupby("channel"):
            _pack(f"channel::{ch}", sub)
    return out


def guess_taxonomy(ref: str, hyp: str, wer: float) -> str:
    """Cheap first-pass labels for the worst-25. A human should overwrite."""
    if not hyp.strip():
        return "truncation_or_empty"
    if len(hyp.split()) > 2 * max(len(ref.split()), 1):
        return "hallucination"
    if has_latin_codeswitch(ref) and not has_latin_codeswitch(hyp):
        return "french_loanword_or_codeswitch"
    r, h = normalize_text(ref), normalize_text(hyp)
    if r == h:
        return "orthographic_variant"
    if wer > 1.0:
        return "hallucination"
    return "needs_human"  # acoustic / NE / bad reference / genuine


def evaluate_manifest(
    kind,
    model,
    processor,
    manifest: Path,
    device: str,
    tag: str,
) -> dict:
    frame = read_jsonl(manifest)
    if "has_cs" not in frame.columns:
        frame["has_cs"] = frame["text"].map(has_latin_codeswitch)
    hyps = []
    audio_s = 0.0
    t0 = time.perf_counter()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=tag):
        wav = load_mono_16k(row.path, SAMPLE_RATE)
        audio_s += float(wav.numel()) / SAMPLE_RATE
        hyps.append(predict_one(kind, model, processor, row.path, device))
    decode_s = time.perf_counter() - t0
    frame = frame.copy()
    frame["hyp"] = hyps
    frame["wer_norm"] = per_utt_wer(frame["text"], frame["hyp"], normalized=True)
    frame["cer_norm"] = per_utt_cer(frame["text"], frame["hyp"], normalized=True)
    frame["wer_raw"] = per_utt_wer(frame["text"], frame["hyp"], normalized=False)
    frame["cer_raw"] = per_utt_cer(frame["text"], frame["hyp"], normalized=False)

    raw = compute_error_rates(frame["text"], frame["hyp"], normalized=False)
    norm = compute_error_rates(frame["text"], frame["hyp"], normalized=True)
    wer_pt, wer_lo, wer_hi = bootstrap_ci(
        frame["text"], frame["hyp"], metric="wer", n_boot=BOOTSTRAP_N, seed=SEED
    )
    cer_pt, cer_lo, cer_hi = bootstrap_ci(
        frame["text"], frame["hyp"], metric="cer", n_boot=BOOTSTRAP_N, seed=SEED
    )

    worst = frame.nlargest(WORST_N, "wer_norm").copy()
    worst["guessed_bucket"] = [
        guess_taxonomy(r, h, w) for r, h, w in zip(worst["text"], worst["hyp"], worst["wer_norm"])
    ]
    worst_path = RESULTS_DIR / f"{tag}_worst{WORST_N}.tsv"
    worst.to_csv(worst_path, sep="\t", index=False)
    details_path = RESULTS_DIR / f"{tag}_details.tsv"
    frame.to_csv(details_path, sep="\t", index=False)

    return {
        "tag": tag,
        "manifest": str(manifest),
        "n": int(len(frame)),
        "hours": float(frame["duration"].sum() / 3600.0),
        "wer_raw": raw.wer,
        "cer_raw": raw.cer,
        "wer_norm": norm.wer,
        "cer_norm": norm.cer,
        "wer_norm_ci95": [wer_lo, wer_pt, wer_hi],
        "cer_norm_ci95": [cer_lo, cer_pt, cer_hi],
        "rtf": decode_s / max(audio_s, 1e-6),
        "slices": slice_table(frame),
        "worst_tsv": str(worst_path),
        "details_tsv": str(details_path),
        "note": (
            "Lead with CER. Darija has no standard orthography; WER punishes "
            "acceptable spelling variants. CIs are utterance-level bootstrap. "
            "On ~400 utts the 95% CI is typically ±2–3 WER points: deltas "
            "inside that band are not interpretable."
        ),
    }


def maybe_teacher_floor() -> dict | None:
    path = Path(__file__).resolve().parents[1] / "data" / "gold_corrections.tsv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    if "human_correction" not in frame.columns:
        return None
    done = frame[frame["human_correction"].fillna("").astype(str).str.strip().ne("")]
    if len(done) < 10:
        return {
            "n_corrected": int(len(done)),
            "status": "incomplete",
            "path": str(path),
        }
    raw = compute_error_rates(done["human_correction"], done["text"], normalized=False)
    norm = compute_error_rates(done["human_correction"], done["text"], normalized=True)
    return {
        "n_corrected": int(len(done)),
        "teacher_wer_raw": raw.wer,
        "teacher_cer_raw": raw.cer,
        "teacher_wer_norm": norm.wer,
        "teacher_cer_norm": norm.cer,
        "interpretation": (
            "This is Gemini vs your corrections. Measured student WER on silver "
            "cannot go below this floor in any meaningful sense."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", default="whisper", choices=("whisper", "ctc", "qwen3_asr", "cohere"))
    parser.add_argument("--name", default="student")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-details", type=Path, default=None)
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    kind, model, processor = load_asr(args.model, args.kind, args.device)
    reports = []
    if not args.skip_silver and (SPLITS_DIR / "silver.jsonl").exists():
        reports.append(
            evaluate_manifest(kind, model, processor, SPLITS_DIR / "silver.jsonl", args.device, f"{args.name}_silver")
        )
    if not args.skip_gold and (SPLITS_DIR / "gold_casablanca.jsonl").exists():
        reports.append(
            evaluate_manifest(
                kind, model, processor, SPLITS_DIR / "gold_casablanca.jsonl", args.device, f"{args.name}_casablanca"
            )
        )
    if (SPLITS_DIR / "gold_atlaset.jsonl").exists():
        reports.append(
            evaluate_manifest(
                kind, model, processor, SPLITS_DIR / "gold_atlaset.jsonl", args.device, f"{args.name}_atlaset"
            )
        )

    paired = None
    if args.baseline_details and args.baseline_details.exists():
        base = pd.read_csv(args.baseline_details, sep="\t")
        student = None
        for rep in reports:
            if Path(rep["details_tsv"]).exists() and "casablanca" in rep["tag"]:
                student = pd.read_csv(rep["details_tsv"], sep="\t")
                break
        if student is not None and len(base) == len(student):
            paired = {
                "wer": paired_bootstrap_delta(student["text"], base["hyp"], student["hyp"], metric="wer"),
                "cer": paired_bootstrap_delta(student["text"], base["hyp"], student["hyp"], metric="cer"),
            }

    payload = {
        "model": args.model,
        "kind": args.kind,
        "name": args.name,
        "single_seed": SEED,
        "teacher_floor": maybe_teacher_floor(),
        "sets": reports,
        "paired_vs_baseline": paired,
    }
    out = RESULTS_DIR / f"{args.name}_eval.json"
    write_json(out, payload)
    print(f"Wrote {out}")
    for rep in reports:
        print(
            f"{rep['tag']}: CER {rep['cer_norm']:.3f} "
            f"[{rep['cer_norm_ci95'][0]:.3f}, {rep['cer_norm_ci95'][2]:.3f}]  "
            f"WER {rep['wer_norm']:.3f}"
        )


if __name__ == "__main__":
    main()
