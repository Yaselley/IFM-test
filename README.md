# Moroccan Darija ASR

Adapt [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) to Moroccan Darija with 3 hours of YouTube audio.

**Train data:** [`01Yassine/darija-asr-3h`](https://huggingface.co/datasets/01Yassine/darija-asr-3h)  
**Eval:** [`atlasia/darija-asr-benchmark`](https://huggingface.co/datasets/atlasia/darija-asr-benchmark) (114 clips, human)

## Open checkpoints

All four recipes are public adapters on the Hub. Inference pulls the Cohere base, then the adapter.

| recipe | Hub | CER | WER |
| --- | --- | ---: | ---: |
| **hybrid** (MultiConv + LoRA) | [`01Yassine/cohere-transcribe-darija`](https://huggingface.co/01Yassine/cohere-transcribe-darija) | **14.4** | **38.3** |
| full LoRA | [`01Yassine/cohere-transcribe-darija-full-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-full-lora) | 16.5 | 40.3 |
| encoder LoRA | [`01Yassine/cohere-transcribe-darija-encoder-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-encoder-lora) | 17.4 | 47.7 |
| decoder LoRA | [`01Yassine/cohere-transcribe-darija-decoder-lora`](https://huggingface.co/01Yassine/cohere-transcribe-darija-decoder-lora) | 20.2 | 45.1 |
| base (no adapter) | [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) | 20.2 | 49.1 |

Writeup: [notebook.ipynb](notebook.ipynb) · [zero-shot](reports/zeroshot.md) · [ablation](reports/finetune.md)

## Inference from Hugging Face

```bash
pip install "transformers>=5.4" peft torch torchaudio soundfile huggingface_hub
```

No local checkpoint needed:

```python
from huggingface_hub import snapshot_download
import sys
sys.path.insert(0, snapshot_download("01Yassine/cohere-transcribe-darija"))
from infer import transcribe
print(transcribe("clip.wav"))
print(transcribe("clip.wav", model_id="full_lora"))
```

From this repo:

```bash
python infer.py clip.wav                      # hybrid
python infer.py clip.wav --model full_lora
python infer.py clip.wav --model encoder_lora
python infer.py clip.wav --model decoder_lora
```

## Train

```bash
export PYTHONPATH=$PWD PYTHONNOUSERSITE=1
python -m src.prepare_data --from-hub
python -m src.train_cohere --recipe method    # hybrid
python scripts/eval_gold.py --model checkpoints/cohere-method/best --name hybrid \
  --out checkpoints/atlasia_eval/method.json
```

Cluster: `bash scripts/srun_train_cohere.sh method`
