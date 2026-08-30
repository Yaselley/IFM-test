# Moroccan Darija ASR

Adapt [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) to Moroccan Darija with 3 hours of YouTube audio.

**Model:** [`01Yassine/cohere-transcribe-darija`](https://huggingface.co/01Yassine/cohere-transcribe-darija)  
**Train data:** [`01Yassine/darija-asr-3h`](https://huggingface.co/datasets/01Yassine/darija-asr-3h)  
**Eval:** [`atlasia/darija-asr-benchmark`](https://huggingface.co/datasets/atlasia/darija-asr-benchmark) (114 clips, human)

| | CER | WER |
| --- | ---: | ---: |
| Cohere Arabic, zero-shot | 20.2 | 49.1 |
| **this adapter (`method`)** | **14.4** | **38.3** |

Full writeup: [notebook.ipynb](notebook.ipynb). Zero-shot lineup: [reports/zeroshot.md](reports/zeroshot.md). Ablation: [reports/finetune.md](reports/finetune.md).

## Try it

Needs `transformers>=5.4`, a GPU helps.

```bash
pip install "transformers>=5.4" peft torch torchaudio soundfile huggingface_hub
python infer.py clip.wav
```

```python
from infer import transcribe
print(transcribe("clip.wav"))
```

That loads the Hub adapter on top of the Cohere base model.

Local checkpoint instead:

```bash
python infer.py clip.wav --model checkpoints/cohere-method/best
```

## Train

```bash
export PYTHONPATH=$PWD PYTHONNOUSERSITE=1
python -m src.prepare_data --from-hub
python -m src.train_cohere --recipe method
python scripts/eval_gold.py --model checkpoints/cohere-method/best --name method \
  --out checkpoints/atlasia_eval/method.json
```

Cluster: `bash scripts/srun_train_cohere.sh method` then `bash scripts/srun_eval_atlasia.sh method`.

Recipes: `method` (conv adapter + decoder LoRA), `full_lora`, `encoder_lora`, `decoder_lora`.
