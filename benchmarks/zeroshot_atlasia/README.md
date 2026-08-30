# Zero-shot on atlasia/darija-asr-benchmark

114 clips, 0.18 h. Human Darija labels from [atlasia/darija-asr-benchmark](https://huggingface.co/datasets/atlasia/darija-asr-benchmark). Gemini never wrote these references.

Same text cleanup for everyone: `src/normalize.py` (tashkeel off, alef/yeh unified, `ة` kept, Latin code-switch kept). Raw and cleaned scores sit next to each other.

CER is the number to read. Darija spelling is not fixed; WER punishes variants a reader would accept.

Notes: `reports/zeroshot.md`.

## Results

| model | params | VRAM MB | RTF | lat ms | CER | WER | CER raw | WER raw | Arabic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper-tiny | 38 | 596 | 0.016 | 96 | 77.0 | 121.0 | 77.6 | 121.6 | 99.1 |
| whisper-base | 73 | 1105 | 0.023 | 132 | 95.5 | 149.6 | 96.7 | 150.7 | 100.0 |
| whisper-small | 242 | 2096 | 0.045 | 265 | 76.9 | 126.0 | 77.4 | 127.4 | 100.0 |
| whisper-medium | 764 | 4441 | 0.118 | 690 | 64.1 | 97.6 | 65.0 | 100.1 | 100.0 |
| whisper-large-v2 | 1543 | 7280 | 0.212 | 1237 | 70.8 | 111.6 | 72.7 | 114.3 | 100.0 |
| whisper-large-v3 | 1544 | 7269 | 0.155 | 905 | 40.9 | 76.6 | 41.8 | 79.6 | 100.0 |
| whisper-large-v3-turbo | 809 | 3888 | 0.041 | 238 | 40.3 | 79.1 | 41.3 | 81.8 | 100.0 |
| qwen3-asr-0.6b | 782 | 2072 | 0.066 | 384 | 32.4 | 76.8 | 33.5 | 79.1 | 100.0 |
| qwen3-asr-1.7b | 2038 | 4490 | 0.103 | 602 | 28.4 | 71.7 | 30.4 | 75.4 | 100.0 |
| moss-transcribe-0.9b | 908 | 2095 | 0.061 | 354 | 95.7 | 99.0 | 96.8 | 99.3 | 5.3 |
| mms-1b-all | 965 | 5029 | 0.017 | 100 | 35.5 | 85.8 | 36.2 | 87.1 | 100.0 |
| cohere-transcribe | 2066 | 4104 | 0.057 | 334 | 43.8 | 77.6 | 44.6 | 80.2 | 100.0 |
| cohere-transcribe-arabic | 2066 | 4103 | 0.078 | 458 | 19.9 | 49.1 | 22.4 | 54.0 | 100.0 |

CER / WER are percentages after our norm. RTF < 1 is faster than real time. Latency is wall time / n (batched models share that cost).

This is [atlasia/darija-asr-benchmark](https://huggingface.co/datasets/atlasia/darija-asr-benchmark): 114 short human-labeled Darija clips. Gemini never wrote these references. n=114 is small; treat gaps under a couple of CER points as noise.

## Why this set

- **whisper**: Whisper — size ladder so we can pick a student that still fits 16GB.
  - `whisper-tiny` — Floor of the Whisper ladder. Cheap. Usually too weak for Darija.
  - `whisper-base` — Still tiny. Checks whether 74M is already useless on Darija.
  - `whisper-small` — Default student. Full FT / LoRA fits on 16GB. Ablation is cheap.
  - `whisper-medium` — Mid-size Whisper. Asks if we pay for medium or jump to turbo/large.
  - `whisper-large-v2` — Large Whisper v2. Size ladder upper end before v3.
  - `whisper-large-v3` — Same paper, stronger zero-shot. Upper bound of stock Whisper.
  - `whisper-large-v3-turbo` — 4-layer decoder. Large-v3 quality at a speed we can actually screen.

- **qwen3-asr**: Qwen3-ASR — 0.6B is a possible student. 1.7B is the stronger multilingual base, zero-shot only in this table.
  - `qwen3-asr-0.6b` — Small Qwen3-ASR. Check if the 0.6B is even usable zero-shot.
  - `qwen3-asr-1.7b` — Qwen3-ASR 1.7B. Strong multilingual ASR; slow vs Cohere on this split.

- **moss**: MOSS-Transcribe-Diarize 0.9B — Whisper-medium encoder + Qwen3-0.6 decoder. Built for long multi-speaker audio. We strip `[S01]` and timestamps before scoring.
  - `moss-transcribe-0.9b` — 0.9B, timestamps + speaker tags. We strip those before WER/CER.

- **wav2vec2**: wav2vec / MMS — CTC. No hallucinated words.
  - `mms-1b-all` — MMS multilingual CTC. Adapter ary, else ara.

- **cohere**: Cohere Transcribe — 2B Conformer. The Arabic finetune is the one that should care about dialects and code-switch. Needs a newer `transformers` than 4.57; if it fails that is why.
  - `cohere-transcribe` — 2B Conformer, 14 languages including Arabic. Needs newer transformers.
  - `cohere-transcribe-arabic` — Same 2B, trained for dialects and Arabic–English code-switch.

## What we left out

- `seamless-m4t-v2-large` (`facebook/seamless-m4t-v2-large`) — Heavy download, different API. Next if time.
- `sensevoice-small` (`FunAudioLLM/SenseVoiceSmall`) — Fast multilingual. FunASR stack, not transformers-native here.

Other names that would be fair later: `FunAudioLLM/SenseVoiceSmall`, `nvidia/canary-1b-flash`, `facebook/seamless-m4t-v2-large`.

## Files

```
results/<model>/hyps.jsonl     # id, ref, ref_norm, hyp, hyp_norm, WER/CER raw+norm, latency
results/<model>/metrics.json   # params, VRAM, RTF, latency, corpus WER/CER
results/<model>/error.txt      # only if it died
```

Run: `python benchmarks/zeroshot_casablanca/run.py --manifest data/splits/gold_atlasia.jsonl --out-dir benchmarks/zeroshot_atlasia/results`
