"""Paths and training defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"

LOCAL_AUDIO_PARQUET = Path(
    "/ds-slt/audio/yelkheir/ITWArabic/sampled_pesq_gt_2.5.parquet"
)
ITWARABIC_ROOT = Path("/ds-slt/audio/yelkheir/ITWArabic")

HF_DATASET = "01Yassine/darija-asr-3h"
CASABLANCA_REPO = "UBC-NLP/Casablanca"
CASABLANCA_CONFIG = "Morocco"
ATLASIA_REPO = "atlasia/darija-asr-benchmark"
ATLASET_REPO = "abdeljalilELmajjodi/Atlaset-audio"

SEED = 42
SAMPLE_RATE = 16000

TRAIN_HOURS = 3.0
DEV_HOURS = 0.15
SILVER_HOURS = 0.35
MIN_DURATION_S = 3.0
MAX_DURATION_S = 15.0
MAX_CHANNEL_SHARE = 0.08

SCREEN_N = 250
GOLD_CORRECTION_N = 50
WORST_N = 25
BOOTSTRAP_N = 1000

COHERE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
COHERE_LANGUAGE = "ar"

COHERE_LORA = dict(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_kinds=("q_proj", "k_proj", "v_proj", "o_proj"),
)

COHERE_ADAPTER = dict(
    bottleneck=64,
    kernels=(7, 15, 23, 31),
    fusion="concat_fusion",
    merge_kernel=31,
    dropout=0.1,
    skip_bottom_frac=0.33,
)

COHERE_TRAIN = dict(
    decoder_lr=1e-4,
    encoder_adapter_lr=1e-4,
    warmup_ratio=0.10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    vram_budget_gb=16.0,
    num_train_epochs=5,
    eval_steps=50,
    logging_steps=10,
    save_total_limit=2,
    max_grad_norm=1.0,
    weight_decay=0.01,
    max_new_tokens=128,
    speed_factors=(0.9, 1.0, 1.1),
    gain_db=3.0,
    drop_chunk_prob=0.3,
    drop_chunk_max_frac=0.15,
    noise_prob=0.25,
    noise_snr=(10.0, 20.0),
    specaug_n_time=2,
    specaug_n_freq=2,
    specaug_time_frac=0.10,
    specaug_freq_frac=0.10,
    musan_prob=0.25,
    rir_prob=0.25,
    musan_root="/ds-slt/audio/MUSAN/musan",
    rir_root="/ds-slt/audio/RIR_Noises/RIRS_NOISES",
)

COHERE_RECIPES = {
    "method": {
        "conv_adapter": True,
        "lora_scope": "decoder",
        "output": "cohere-method",
        "why": "MultiConvAdapter on the frozen Conformer + decoder LoRA",
    },
    "decoder_lora": {
        "conv_adapter": False,
        "lora_scope": "decoder",
        "output": "cohere-decoder-lora",
        "why": "Frozen encoder. Writing convention only.",
    },
    "full_lora": {
        "conv_adapter": False,
        "lora_scope": "full",
        "output": "cohere-full-lora",
        "why": "Encoder+decoder LoRA, no conv adapter. Standard PEFT.",
    },
    "encoder_lora": {
        "conv_adapter": False,
        "lora_scope": "encoder",
        "output": "cohere-encoder-lora",
        "why": "Encoder LoRA, frozen decoder. Is the gain acoustic?",
    },
}

WHISPER_LANGUAGE = "arabic"
WHISPER_TASK = "transcribe"

CASABLANCA_PUBLISHED = {
    "whisper-large-v2": {"wer": 88.55, "cer": 46.57},
    "whisper-large-v3": {"wer": 84.52, "cer": 44.02},
    "seamless-m4t-v2-large": {"wer": 87.2, "cer": 44.41},
    "mms-1b-all": {"wer": 83.05, "cer": 42.09},
}


def ensure_dirs() -> None:
    for path in (SPLITS_DIR, RESULTS_DIR, CHECKPOINTS_DIR, DATA_DIR / "cache"):
        path.mkdir(parents=True, exist_ok=True)
