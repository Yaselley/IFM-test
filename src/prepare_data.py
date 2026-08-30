"""Build the 3h channel-disjoint train/dev/silver set + Casablanca gold."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm

from src.config import (
    ATLASET_REPO,
    CASABLANCA_CONFIG,
    CASABLANCA_REPO,
    DATA_DIR,
    DEV_HOURS,
    GOLD_CORRECTION_N,
    ITWARABIC_ROOT,
    LOCAL_AUDIO_PARQUET,
    MAX_CHANNEL_SHARE,
    MAX_DURATION_S,
    MIN_DURATION_S,
    SAMPLE_RATE,
    SCREEN_N,
    SEED,
    SILVER_HOURS,
    SPLITS_DIR,
    TRAIN_HOURS,
    HF_DATASET,
    ensure_dirs,
)
from src.normalize import has_latin_codeswitch

_VIDEO_RE = re.compile(r"(\d+)_audio")


def hub_id_from_path(path: str) -> str | None:
    parts = Path(path).parts
    try:
        root_idx = parts.index("ITWArabic")
    except ValueError:
        return None
    channel = parts[root_idx + 1]
    video_dir = parts[-2]
    match = _VIDEO_RE.search(video_dir)
    if match is None:
        return None
    return f"{channel}_{match.group(1)}_{Path(path).stem}"


def channel_from_path(path: str) -> str | None:
    parts = Path(path).parts
    try:
        root_idx = parts.index("ITWArabic")
    except ValueError:
        return None
    return parts[root_idx + 1]


def _texts_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    id_col = "file_id" if "file_id" in frame.columns else "id"
    text_col = "transcription" if "transcription" in frame.columns else "text"
    out = frame[[id_col, text_col]].rename(columns={id_col: "hub_id", text_col: "text"})
    out["text"] = out["text"].fillna("").astype(str)
    return out[out["text"].str.strip().ne("")]


def load_transcripts() -> pd.DataFrame:
    """Gemini labels from the published 3h set (all splits)."""
    frames = []
    for split in ("train", "validation", "silver"):
        ds = load_dataset(HF_DATASET, split=split)
        drop = [c for c in ds.column_names if c == "audio"]
        if drop:
            ds = ds.remove_columns(drop)
        frames.append(_texts_from_frame(ds.to_pandas()))
    return pd.concat(frames, ignore_index=True).drop_duplicates("hub_id")


def materialize_from_hub() -> dict[str, pd.DataFrame]:
    """Download 01Yassine/darija-asr-3h and write local wavs + jsonl."""
    import soundfile as sf

    hub_map = {"train": "train", "dev": "validation", "silver": "silver"}
    out: dict[str, pd.DataFrame] = {}
    for local_name, hub_split in hub_map.items():
        ds = load_dataset(HF_DATASET, split=hub_split)
        cache = DATA_DIR / "cache" / "darija-asr-3h" / local_name
        cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, ex in enumerate(tqdm(ds, desc=f"hub/{local_name}")):
            uid = str(ex.get("id") or f"{local_name}_{i:05d}")
            wav_path = cache / f"{uid.replace('/', '_')}.wav"
            audio = ex["audio"]
            if not wav_path.exists():
                sf.write(
                    str(wav_path),
                    audio["array"],
                    int(audio.get("sampling_rate") or SAMPLE_RATE),
                )
            text = ex.get("text") or ""
            if not isinstance(text, str):
                text = ""
            rows.append(
                {
                    "id": uid,
                    "path": str(wav_path),
                    "text": text,
                    "channel": ex.get("channel"),
                    "duration": float(ex.get("duration") or 0.0),
                    "pesq_hyp": ex.get("pesq_hyp"),
                    "num_speakers": ex.get("num_speakers"),
                    "has_cs": bool(
                        ex.get("has_cs")
                        if ex.get("has_cs") is not None
                        else has_latin_codeswitch(text)
                    ),
                    "source": "gemini",
                    "split": local_name,
                }
            )
        out[local_name] = pd.DataFrame(rows)
    return out


def load_local_audio_table() -> pd.DataFrame:
    audio = pd.read_parquet(LOCAL_AUDIO_PARQUET)
    audio["hub_id"] = audio["path"].map(hub_id_from_path)
    audio["channel"] = audio["path"].map(channel_from_path)
    audio = audio.dropna(subset=["hub_id", "channel", "path"])
    # Speaker parquet is 342k rows on NFS — skip unless the caller asks.
    if "duration" not in audio.columns:
        audio["duration"] = np.nan
    if "num_speakers" not in audio.columns:
        audio["num_speakers"] = np.nan
    return audio


def fill_durations(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer filesize (one stat) over opening WAV headers on NFS."""
    missing = frame["duration"].isna() | (frame["duration"] <= 0)
    if not missing.any():
        return frame

    def _from_stat(path: str) -> float:
        try:
            size = Path(path).stat().st_size
        except OSError:
            return float("nan")
        return max(size - 44, 0) / float(SAMPLE_RATE * 2)

    frame.loc[missing, "duration"] = [
        _from_stat(p) for p in tqdm(frame.loc[missing, "path"], desc="duration(stat)")
    ]
    return frame


def subsample_hours(
    frame: pd.DataFrame,
    hours: float,
    rng: np.random.Generator,
    *,
    channel_cap: float | None = None,
) -> pd.DataFrame:
    if frame.empty or hours <= 0:
        return frame.iloc[0:0].copy()
    work = frame.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).copy()
    if channel_cap is not None:
        budget_s = hours * 3600.0
        kept = []
        used_by_channel: dict[str, float] = {}
        total = 0.0
        for row in work.itertuples(index=False):
            ch = row.channel
            already = used_by_channel.get(ch, 0.0)
            if already >= channel_cap * budget_s:
                continue
            kept.append(row)
            used_by_channel[ch] = already + float(row.duration)
            total += float(row.duration)
            if total >= budget_s:
                break
        return pd.DataFrame(kept)
    target = hours * 3600.0
    work["cum"] = work["duration"].cumsum()
    return work[work["cum"] - work["duration"] < target].drop(columns=["cum"])


def _dump_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = frame.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _text_column(ds) -> str:
    for name in ("transcription", "sentence", "text", "darija_ar", "transcript"):
        if name in ds.column_names:
            return name
    raise KeyError(f"no text column in {ds.column_names}")


def _casablanca_local_parquets() -> list[tuple[str, Path]]:
    """Use Hub-cache Morocco shards when present (no extra download)."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    morocco = hub / "datasets--UBC-NLP--Casablanca" / "snapshots"
    found: list[tuple[str, Path]] = []
    if not morocco.exists():
        return found
    for snap in morocco.iterdir():
        split_dir = snap / CASABLANCA_CONFIG
        if not split_dir.is_dir():
            continue
        for split in ("test", "validation"):
            found.extend((split, p) for p in sorted(split_dir.glob(f"{split}-*.parquet")))
        if found:
            return found
    return found


def _prepare_casablanca_from_parquets(
    shards: list[tuple[str, Path]],
    max_per_split: int | None = None,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    rows = []
    counts: dict[str, int] = {}
    for split, parquet_path in shards:
        if max_per_split is not None and counts.get(split, 0) >= max_per_split:
            continue
        cache_dir = DATA_DIR / "cache" / "casablanca" / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(parquet_path)
        for batch in tqdm(
            pf.iter_batches(batch_size=16),
            desc=f"casablanca/{split}/{parquet_path.name}",
            total=max(1, pf.metadata.num_rows // 16),
        ):
            data = batch.to_pydict()
            for audio, text, duration, gender in zip(
                data["audio"],
                data["transcription"],
                data["duration"],
                data["gender"],
            ):
                i = counts.get(split, 0)
                if max_per_split is not None and i >= max_per_split:
                    break
                wav_path = cache_dir / f"{split}_{i:05d}.wav"
                blob = audio.get("bytes") if isinstance(audio, dict) else None
                if blob and not wav_path.exists():
                    wav_path.write_bytes(blob)
                text_s = text if isinstance(text, str) else ""
                rows.append(
                    {
                        "id": f"casablanca_{split}_{i:05d}",
                        "path": str(wav_path),
                        "text": text_s,
                        "channel": "casablanca_morocco",
                        "duration": float(duration),
                        "pesq_hyp": np.nan,
                        "num_speakers": 1,
                        "has_cs": has_latin_codeswitch(text_s),
                        "split": f"gold_{split}",
                        "source": "casablanca",
                        "gender": gender,
                    }
                )
                counts[split] = i + 1
    return pd.DataFrame(rows)


def prepare_casablanca(max_per_split: int | None = None) -> pd.DataFrame:
    """Prefer local Hub-cache parquets; stream from the Hub only if needed."""
    local = _casablanca_local_parquets()
    if local:
        print(f"Casablanca: {len(local)} local parquet shards", flush=True)
        return _prepare_casablanca_from_parquets(local, max_per_split)

    import soundfile as sf

    rows = []
    for split in ("test", "validation"):
        try:
            ds = load_dataset(
                CASABLANCA_REPO,
                CASABLANCA_CONFIG,
                split=split,
                streaming=True,
            )
        except Exception as exc:
            print(f"[warn] Casablanca {split}: {exc}", flush=True)
            continue
        text_col = _text_column(ds)
        cache_dir = DATA_DIR / "cache" / "casablanca" / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(tqdm(ds, desc=f"casablanca/{split}")):
            audio = ex["audio"]
            array = audio["array"]
            sr = int(audio.get("sampling_rate") or SAMPLE_RATE)
            wav_path = cache_dir / f"{split}_{i:05d}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), array, sr)
            text = ex[text_col]
            duration = ex.get("duration")
            if duration is None:
                duration = len(array) / float(sr)
            rows.append(
                {
                    "id": f"casablanca_{split}_{i:05d}",
                    "path": str(wav_path),
                    "text": text if isinstance(text, str) else "",
                    "channel": "casablanca_morocco",
                    "duration": float(duration),
                    "pesq_hyp": np.nan,
                    "num_speakers": 1,
                    "has_cs": bool(ex.get("code_switching") or has_latin_codeswitch(text)),
                    "split": f"gold_{split}",
                    "source": "casablanca",
                    "gender": ex.get("gender"),
                }
            )
            del array, audio, ex
            if max_per_split is not None and (i + 1) >= max_per_split:
                break
    return pd.DataFrame(rows)


def prepare_atlaset() -> pd.DataFrame:
    try:
        ds = load_dataset(ATLASET_REPO, split="train")
    except Exception as exc:
        print(f"[warn] Atlaset-audio unavailable: {exc}")
        return pd.DataFrame()
    text_col = _text_column(ds)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE)) if "audio" in ds.column_names else ds
    cache_dir = DATA_DIR / "cache" / "atlaset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(len(ds), 101)
    for i in tqdm(range(n), desc="atlaset"):
        ex = ds[i]
        audio = ex.get("audio")
        if not audio:
            continue
        wav_path = cache_dir / f"atlaset_{i:03d}.wav"
        if not wav_path.exists():
            import soundfile as sf

            sf.write(str(wav_path), audio["array"], audio.get("sampling_rate", SAMPLE_RATE))
        text = ex.get(text_col) or ex.get("darija_ar") or ""
        rows.append(
            {
                "id": f"atlaset_{i:03d}",
                "path": str(wav_path),
                "text": text if isinstance(text, str) else "",
                "channel": "atlaset",
                "duration": float(len(audio["array"]) / audio.get("sampling_rate", SAMPLE_RATE)),
                "pesq_hyp": np.nan,
                "num_speakers": 1,
                "has_cs": has_latin_codeswitch(text),
                "split": "gold_atlaset",
                "source": "atlaset",
            }
        )
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "hours": 0.0, "n_channels": 0}
    return {
        "n": int(len(frame)),
        "hours": float(frame["duration"].sum() / 3600.0),
        "n_channels": int(frame["channel"].nunique()),
        "mean_duration_s": float(frame["duration"].mean()),
        "whisper_pad_waste": float(1.0 - frame["duration"].mean() / 30.0),
        "cs_rate": float(frame["has_cs"].mean()) if "has_cs" in frame else None,
        "mean_pesq": float(frame["pesq_hyp"].mean()) if frame["pesq_hyp"].notna().any() else None,
    }


def build_student_splits(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    print("Loading local audio metadata…")
    audio = load_local_audio_table()
    print(f"  {len(audio)} rows, {audio['channel'].nunique()} channels")
    print("Loading Gemini transcripts…")
    transcripts = load_transcripts()
    print(f"  {len(transcripts)} non-empty transcripts")
    merged = audio.merge(transcripts, on="hub_id", how="inner")
    print(f"  joined {len(merged)} rows", flush=True)
    # Duration uses filesize, not wave.open, so we do not hit NFS headers.
    merged = fill_durations(merged)
    merged = merged[
        merged["duration"].between(MIN_DURATION_S, MAX_DURATION_S, inclusive="both")
    ].copy()
    merged["has_cs"] = merged["text"].map(has_latin_codeswitch)
    merged["id"] = merged["hub_id"]
    merged["source"] = "gemini"
    merged["num_speakers"] = 1

    cols = [
        "id",
        "path",
        "text",
        "channel",
        "duration",
        "pesq_hyp",
        "num_speakers",
        "has_cs",
        "source",
    ]
    pool = merged[cols].drop_duplicates("id")
    print(f"  filtered pool: {len(pool)} utts / {pool['duration'].sum()/3600:.2f}h")

    channels = pool["channel"].unique()
    rng.shuffle(channels)
    n_hold = max(8, int(0.15 * len(channels)))
    hold = set(channels[:n_hold])
    silver_pool = pool[pool["channel"].isin(hold)]
    train_pool = pool[~pool["channel"].isin(hold)]

    silver = subsample_hours(silver_pool, SILVER_HOURS, rng)
    train = subsample_hours(train_pool, TRAIN_HOURS, rng, channel_cap=MAX_CHANNEL_SHARE)
    leftover = train_pool[~train_pool["id"].isin(set(train["id"]))]
    dev = subsample_hours(leftover, DEV_HOURS, rng)

    # Same-size random split on the same pool, to quantify leakage.
    shuffled = pool.sample(frac=1.0, random_state=SEED)
    train_rand = subsample_hours(shuffled, TRAIN_HOURS, rng)
    rest = shuffled[~shuffled["id"].isin(set(train_rand["id"]))]
    dev_rand = subsample_hours(rest, DEV_HOURS, rng)
    rest2 = rest[~rest["id"].isin(set(dev_rand["id"]))]
    silver_rand = subsample_hours(rest2, SILVER_HOURS, rng)

    leak_channels = set(train_rand["channel"]) & set(silver_rand["channel"])
    leak_frac = (
        silver_rand["channel"].isin(leak_channels).mean() if len(silver_rand) else 0.0
    )

    train = train.assign(split="train")
    dev = dev.assign(split="dev")
    silver = silver.assign(split="silver")
    train_rand = train_rand.assign(split="train_random")
    dev_rand = dev_rand.assign(split="dev_random")
    silver_rand = silver_rand.assign(split="silver_random")

    stats = {
        "pipeline": (
            "YouTube crawl → Silero VAD → SQUIM/PESQ → DNS64 → pyannote → "
            "Gemini 2.5 Pro. 3h channel-disjoint, pesq_hyp > 2.5. "
            f"Hub: {HF_DATASET}."
        ),
        "label_provenance": (
            "Gemini 2.5 Pro transcripts I generated for this crawl. "
            "Silver WER is agreement with those labels, not a human ref."
        ),
        "filters": {
            "min_duration_s": MIN_DURATION_S,
            "max_duration_s": MAX_DURATION_S,
            "pesq": "pesq_hyp > 2.5 — train is clean-audio best case",
            "max_channel_share": MAX_CHANNEL_SHARE,
        },
        "channel_disjoint": {
            "held_out_channels": sorted(hold),
            "n_held_out_channels": len(hold),
            "train": _summary(train),
            "dev": _summary(dev),
            "silver": _summary(silver),
        },
        "random_split": {
            "train": _summary(train_rand),
            "dev": _summary(dev_rand),
            "silver": _summary(silver_rand),
            "silver_utts_sharing_a_train_channel": float(leak_frac),
            "n_leaking_channels": len(leak_channels),
        },
        "whisper_padding": {
            "note": "Whisper pads every clip to 30s. Mean duration / 30 is the useful fraction.",
            "train_useful_frac": float(train["duration"].mean() / 30.0) if len(train) else None,
        },
        "pool": _summary(pool),
        "local_audio_root": str(ITWARABIC_ROOT),
    }
    return {
        "train": train,
        "dev": dev,
        "silver": silver,
        "train_random": train_rand,
        "dev_random": dev_rand,
        "silver_random": silver_rand,
        "pool": pool,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-casablanca", action="store_true")
    parser.add_argument("--skip-atlaset", action="store_true")
    parser.add_argument("--casablanca-only", action="store_true")
    parser.add_argument("--from-hub", action="store_true", help=f"Write splits from {HF_DATASET}")
    parser.add_argument("--rebuild", action="store_true", help="Re-cut 3h from the local processed pool")
    parser.add_argument("--casablanca-max-per-split", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    if args.casablanca_only:
        casa = prepare_casablanca(max_per_split=args.casablanca_max_per_split)
        if casa.empty:
            raise SystemExit("Casablanca download returned no rows")
        _dump_jsonl(casa, SPLITS_DIR / "gold_casablanca.jsonl")
        test = casa[casa["split"] == "gold_test"]
        if test.empty:
            test = casa
        screen = test.sample(n=min(SCREEN_N, len(test)), random_state=SEED)
        _dump_jsonl(screen, SPLITS_DIR / "screen.jsonl")
        print(json.dumps({"casablanca": _summary(casa), "screen": _summary(screen)}, indent=2))
        return

    have_local = (SPLITS_DIR / "train.jsonl").exists()
    if args.rebuild:
        splits = build_student_splits(rng)
    elif args.from_hub or not have_local:
        hub = materialize_from_hub()
        splits = {
            "train": hub["train"],
            "dev": hub["dev"],
            "silver": hub["silver"],
            "stats": {
                "pipeline": (
                    "YouTube crawl → Silero VAD → SQUIM/PESQ → DNS64 → pyannote → "
                    f"Gemini 2.5 Pro. Hub: {HF_DATASET}."
                ),
                "label_provenance": (
                    "Gemini 2.5 Pro transcripts I generated for this crawl. "
                    "Silver WER is agreement with those labels, not a human ref."
                ),
                "hub": HF_DATASET,
                "channel_disjoint": {
                    "train": _summary(hub["train"]),
                    "dev": _summary(hub["dev"]),
                    "silver": _summary(hub["silver"]),
                },
            },
        }
    else:
        splits = {
            "train": pd.read_json(SPLITS_DIR / "train.jsonl", lines=True),
            "dev": pd.read_json(SPLITS_DIR / "dev.jsonl", lines=True),
            "silver": pd.read_json(SPLITS_DIR / "silver.jsonl", lines=True),
            "stats": json.loads((SPLITS_DIR / "stats.json").read_text())
            if (SPLITS_DIR / "stats.json").exists()
            else {},
        }
    for name in ("train", "dev", "silver"):
        _dump_jsonl(splits[name], SPLITS_DIR / f"{name}.jsonl")

    gold_frames = []
    if not args.skip_casablanca:
        casa = prepare_casablanca(max_per_split=args.casablanca_max_per_split)
        if not casa.empty:
            _dump_jsonl(casa, SPLITS_DIR / "gold_casablanca.jsonl")
            gold_frames.append(casa)
            test = casa[casa["split"] == "gold_test"]
            if test.empty:
                test = casa
            screen = test.sample(n=min(SCREEN_N, len(test)), random_state=SEED)
            _dump_jsonl(screen, SPLITS_DIR / "screen.jsonl")
            splits["stats"]["casablanca"] = _summary(casa)
            splits["stats"]["screen"] = _summary(screen)

    if not args.skip_atlaset:
        atlaset = prepare_atlaset()
        if not atlaset.empty:
            _dump_jsonl(atlaset, SPLITS_DIR / "gold_atlaset.jsonl")
            splits["stats"]["atlaset"] = _summary(atlaset)

    silver = splits["silver"]
    corrections = silver.sample(n=min(GOLD_CORRECTION_N, len(silver)), random_state=SEED)
    corr_path = DATA_DIR / "gold_corrections.tsv"
    corrections.assign(human_correction="", notes="").to_csv(
        corr_path,
        sep="\t",
        index=False,
        columns=["id", "path", "text", "channel", "duration", "human_correction", "notes"],
    )
    splits["stats"]["gold_corrections"] = {
        "path": str(corr_path),
        "n": int(len(corrections)),
        "instruction": (
            "Hand-correct the Gemini `text` column. Teacher WER against your "
            "corrections is the label-noise floor. Differences below that "
            "floor are not interpretable."
        ),
    }

    stats_path = SPLITS_DIR / "stats.json"
    stats_path.write_text(json.dumps(splits["stats"], indent=2, ensure_ascii=False))
    print(json.dumps(splits["stats"], indent=2, ensure_ascii=False))
    print(f"Wrote splits to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
