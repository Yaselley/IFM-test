# Zero-shot atlasia/darija-asr-benchmark

114 clips, 0.18 h, human Darija labels from [`atlasia/darija-asr-benchmark`](https://huggingface.co/datasets/atlasia/darija-asr-benchmark). Same `src/normalize.py` for everyone. **CER** is the number that matters, Darija spelling is not fixed. n=114 is small; a couple of CER points is noise.

| model | params | CER | WER | RTF | lat ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| whisper-tiny | 38 | 77.0 | 121.0 | 0.016 | 96 |
| whisper-base | 73 | 95.5 | 149.6 | 0.023 | 132 |
| whisper-small | 242 | 76.9 | 126.0 | 0.045 | 265 |
| whisper-medium | 764 | 64.1 | 97.6 | 0.118 | 690 |
| whisper-large-v2 | 1543 | 70.8 | 111.6 | 0.212 | 1237 |
| whisper-large-v3 | 1544 | 40.9 | 76.6 | 0.155 | 905 |
| whisper-large-v3-turbo | 809 | 40.3 | 79.1 | 0.041 | 238 |
| qwen3-asr-0.6b | 782 | 32.4 | 76.8 | 0.066 | 384 |
| qwen3-asr-1.7b | 2038 | 28.4 | 71.7 | 0.103 | 602 |
| moss-transcribe-0.9b | 908 | 95.7 | 99.0 | 0.061 | 354 |
| mms-1b-all | 965 | 35.5 | 85.8 | 0.017 | 100 |
| cohere-transcribe | 2066 | 43.8 | 77.6 | 0.057 | 334 |
| **cohere-transcribe-arabic** | **2066** | **19.9** | **49.1** | **0.078** | **458** |

I picked the Arabic Cohere card. Same size as multilingual Cohere (43.8 CER), much better on Darija. Qwen 1.7B is the one I would take if I needed an LLM-first fallback.

Adapted checkpoints (all public):

| recipe | Hub |
| --- | --- |
| hybrid | [`01Yassine/cohere-transcribe-darija`](https://huggingface.co/01Yassine/cohere-transcribe-darija) |
| full LoRA | [`01Yassine/cohere-transcribe-darija-full-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-full-lora) |
| encoder LoRA | [`01Yassine/cohere-transcribe-darija-encoder-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-encoder-lora) |
| decoder LoRA | [`01Yassine/cohere-transcribe-darija-decoder-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-decoder-lora) |

`python infer.py clip.wav --model hybrid`

MOSS is 5% Arabic script; its CER is not a transcription number. Whisper is not a starting point.

Full per-clip files: `benchmarks/zeroshot_atlasia/results/`. Rerun: `bash scripts/srun_zeroshot_atlasia.sh`.
