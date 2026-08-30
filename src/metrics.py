"""WER / CER, bootstrap CIs, tokenizer fertility, RTF."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from jiwer import cer as jiwer_cer
from jiwer import process_characters, process_words
from jiwer import wer as jiwer_wer

from src.normalize import normalize_text


@dataclass
class ErrorRates:
    wer: float
    cer: float
    n_utts: int
    n_words: int
    n_chars: int
    insertions: int
    deletions: int
    substitutions: int


def _safe_join(texts: Sequence[str]) -> list[str]:
    return [t if t.strip() else "<empty>" for t in texts]


def compute_error_rates(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalized: bool = True,
) -> ErrorRates:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same length")
    if normalized:
        refs = [normalize_text(r) for r in references]
        hyps = [normalize_text(h) for h in hypotheses]
    else:
        refs = [str(r or "") for r in references]
        hyps = [str(h or "") for h in hypotheses]
    refs_s = _safe_join(refs)
    hyps_s = _safe_join(hyps)
    words = process_words(refs_s, hyps_s)
    chars = process_characters(refs_s, hyps_s)
    n_words = max(int(words.hits + words.substitutions + words.deletions), 1)
    n_chars = max(int(chars.hits + chars.substitutions + chars.deletions), 1)
    return ErrorRates(
        wer=float(jiwer_wer(refs_s, hyps_s)),
        cer=float(jiwer_cer(refs_s, hyps_s)),
        n_utts=len(refs),
        n_words=n_words,
        n_chars=n_chars,
        insertions=int(words.insertions),
        deletions=int(words.deletions),
        substitutions=int(words.substitutions),
    )


def per_utt_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalized: bool = True,
) -> list[float]:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        r = normalize_text(ref) if normalized else str(ref or "")
        h = normalize_text(hyp) if normalized else str(hyp or "")
        scores.append(float(jiwer_wer(_safe_join([r]), _safe_join([h]))))
    return scores


def per_utt_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalized: bool = True,
) -> list[float]:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        r = normalize_text(ref) if normalized else str(ref or "")
        h = normalize_text(hyp) if normalized else str(hyp or "")
        scores.append(float(jiwer_cer(_safe_join([r]), _safe_join([h]))))
    return scores


def bootstrap_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    metric: str = "cer",
    n_boot: int = 1000,
    seed: int = 42,
    normalized: bool = True,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a 95% utterance-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(references)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    scores = (
        per_utt_cer(references, hypotheses, normalized=normalized)
        if metric == "cer"
        else per_utt_wer(references, hypotheses, normalized=normalized)
    )
    scores = np.asarray(scores, dtype=np.float64)
    point = float(scores.mean())
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = scores[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def paired_bootstrap_delta(
    references: Sequence[str],
    hyp_a: Sequence[str],
    hyp_b: Sequence[str],
    *,
    metric: str = "cer",
    n_boot: int = 1000,
    seed: int = 42,
    normalized: bool = True,
) -> dict[str, float]:
    """Bootstrap the mean(A) - mean(B) difference. Positive => A is worse."""
    rng = np.random.default_rng(seed)
    scorer = per_utt_cer if metric == "cer" else per_utt_wer
    a = np.asarray(scorer(references, hyp_a, normalized=normalized), dtype=np.float64)
    b = np.asarray(scorer(references, hyp_b, normalized=normalized), dtype=np.float64)
    n = len(a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(a.mean() - b.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "significant": not (lo <= 0.0 <= hi),
        "n": float(n),
    }


def tokenizer_fertility(tokenizer, texts: Iterable[str]) -> dict[str, float]:
    """Tokens per whitespace word. High fertility = more steps, harder seq2seq."""
    words = 0
    tokens = 0
    n = 0
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        n_words = max(len(text.split()), 1)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        words += n_words
        tokens += len(ids)
        n += 1
    return {
        "tokens_per_word": tokens / max(words, 1),
        "n_utts": n,
        "n_words": words,
        "n_tokens": tokens,
    }
