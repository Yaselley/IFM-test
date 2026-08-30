# Fine-tune

Student on the Hub: [`01Yassine/cohere-transcribe-darija`](https://huggingface.co/01Yassine/cohere-transcribe-darija). `python infer.py clip.wav`

Student: `CohereLabs/cohere-transcribe-arabic-07-2026`. Freeze the Conformer. Train a few million params on 3h Gemini labels. Keep-best on **dev CER**. The number to report is gold eval on [`atlasia/darija-asr-benchmark`](https://huggingface.co/datasets/atlasia/darija-asr-benchmark) (114 clips, human).

## Why this recipe

- Zero-shot is already 19.9 CER on AtlasIA (20.2 on the student eval script). A high LR wipes that. Adapters 1e-4, decoder LoRA 5e-4. Not 1e-3.
- Mismatch is temporal (rate, gemination, French bursts) → conv adapter, not LoRA on the Conformer. Kernels 7/15/23/31 from MULTI-CONVFORMER. Skip the bottom third of layers.
- Decoder LoRA r=32 on q/k/v/o (self-attn + cross-attn). No MLP, no embed — 3h of Gemini overfits spelling there.
- Train audio is PESQ > 2.5. AtlasIA is not. Keep SpeechBrain-style speed/gain/drop + SpecAugment on every clip. Add MUSAN and RIR at 25%. No MUSAN speech.
- Mean clip is 6s. Random batches waste ~50% on pad. Length buckets drop that to ~5–11%.
- Recipe sized for 16GB. Eval generate stays batch 1.

## Ablation

Same data, seed, buckets, aug. Only the trainable slice changes.

| `--recipe` | encoder | decoder |
| --- | --- | --- |
| `method` | MultiConvAdapter | LoRA |
| `decoder_lora` | frozen | LoRA |
| `full_lora` | LoRA | LoRA |
| `encoder_lora` | LoRA | frozen |

If `method` beats `full_lora`, the kernels did something LoRA did not. If `encoder_lora` matches `method`, the decoder was not the story.

## Validation (Gemini, 91 clips)

Keep-best on **dev CER**. Teacher agreement, not gold.

| recipe | CER | CER (norm) | WER | WER (norm) | trainable |
| --- | ---: | ---: | ---: | ---: | ---: |
| **method** | 15.6 | **14.9** | 34.4 | **31.8** | 21.1M |
| **full_lora** | 15.6 | **14.9** | 34.7 | **32.4** | 19.9M |
| encoder_lora | 19.3 | 17.4 | 42.2 | 38.1 | 15.7M |
| decoder_lora | 18.6 | 17.8 | 42.2 | 39.3 | 4.2M |

`method` and `full_lora` tie here. 91 clips cannot call a winner.

## Gold: AtlasIA (human, 114 clips)

Same checkpoints, `data/splits/gold_atlasia.jsonl`.

| recipe | CER | CER (norm) | WER | WER (norm) |
| --- | ---: | ---: | ---: | ---: |
| zero-shot | 22.7 | **20.2** | 53.9 | **49.1** |
| **method** | 15.3 | **14.4** | 42.2 | **38.3** |
| full_lora | 17.3 | 16.5 | 43.8 | 40.3 |
| encoder_lora | 19.5 | 17.4 | 52.3 | 47.7 |
| decoder_lora | 21.2 | 20.2 | 48.6 | 45.1 |

On the human set the tie breaks: **`method` is the student**, 20.2 → 14.4 CER. Encoder-only and decoder-only barely move the baseline. n=114 is small; a couple of CER points is the noise floor.
