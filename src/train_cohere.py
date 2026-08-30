"""Fine-tune Cohere Transcribe Arabic.

Freeze the 2B Conformer. Train decoder LoRA and (recipe=method) MultiConvAdapter.
See reports/finetune.md.

    python -m src.train_cohere --recipe method
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from src.augment import WaveformAugment, spec_augment_mel
from src.cohere_runtime import (
    attach_decoder_lora,
    load_cohere_base,
    prepare_student,
    save_student,
)
from src.config import (
    CHECKPOINTS_DIR,
    COHERE_ADAPTER,
    COHERE_LANGUAGE,
    COHERE_LORA,
    COHERE_MODEL,
    COHERE_RECIPES,
    COHERE_TRAIN,
    DATA_DIR,
    SAMPLE_RATE,
    SEED,
    SPLITS_DIR,
    ensure_dirs,
)
from src.io_utils import load_mono_16k, read_jsonl, write_json
from src.metrics import compute_error_rates


def _pad_features(input_features: list[torch.Tensor], attention_mask: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    max_t = max(feat.size(0) for feat in input_features)
    feat_dim = input_features[0].size(-1)
    feats = input_features[0].new_zeros(len(input_features), max_t, feat_dim)
    mask = torch.zeros(len(input_features), max_t, dtype=torch.long)
    for i, (feat, att) in enumerate(zip(input_features, attention_mask)):
        t = feat.size(0)
        feats[i, :t] = feat
        mask[i, : min(t, att.numel())] = att[:t].to(dtype=torch.long)
    return {"input_features": feats, "attention_mask": mask}


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


@dataclass
class AsrItem:
    path: str
    text: str
    duration: float = 0.0


class CohereJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, processor, augment: bool, seed: int):
        frame = read_jsonl(manifest)
        durations = frame["duration"] if "duration" in frame.columns else [0.0] * len(frame)
        self.items = [
            AsrItem(p, t, float(d) if d is not None else 0.0)
            for p, t, d in zip(frame["path"], frame["text"], durations)
        ]
        self.processor = processor
        self.prompt_ids = list(processor.get_decoder_prompt_ids(language=COHERE_LANGUAGE, punctuation=True))
        self.eos_id = processor.tokenizer.eos_token_id
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.wave_aug = (
            WaveformAugment(
                sample_rate=SAMPLE_RATE,
                speed_factors=COHERE_TRAIN["speed_factors"],
                gain_db=COHERE_TRAIN["gain_db"],
                drop_chunk_prob=COHERE_TRAIN["drop_chunk_prob"],
                drop_chunk_max_frac=COHERE_TRAIN["drop_chunk_max_frac"],
                noise_prob=COHERE_TRAIN["noise_prob"],
                noise_snr=COHERE_TRAIN["noise_snr"],
                seed=seed,
                rir_prob=COHERE_TRAIN["rir_prob"],
                musan_prob=COHERE_TRAIN["musan_prob"],
                musan_root=COHERE_TRAIN["musan_root"],
                rir_root=COHERE_TRAIN["rir_root"],
                cache_dir=DATA_DIR / "cache",
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def _label_ids(self, text: str) -> list[int]:
        text_ids = self.processor.tokenizer(text, add_special_tokens=False).input_ids
        full = list(self.prompt_ids) + list(text_ids)
        if self.eos_id is not None and (not full or full[-1] != self.eos_id):
            full.append(self.eos_id)
        return full

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav = load_mono_16k(item.path, SAMPLE_RATE)
        if self.wave_aug is not None:
            wav = self.wave_aug(wav)
        feats = self.processor.feature_extractor(
            wav.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feats["input_features"].squeeze(0)
        attention_mask = feats["attention_mask"].squeeze(0)
        if self.augment:
            input_features = spec_augment_mel(
                input_features,
                n_time=COHERE_TRAIN["specaug_n_time"],
                n_freq=COHERE_TRAIN["specaug_n_freq"],
                time_frac=COHERE_TRAIN["specaug_time_frac"],
                freq_frac=COHERE_TRAIN["specaug_freq_frac"],
                rng=self.rng,
            )
            input_features = input_features * attention_mask.to(input_features.dtype).unsqueeze(-1)
        labels = torch.tensor(self._label_ids(item.text), dtype=torch.long)
        prompt = torch.tensor(self.prompt_ids, dtype=torch.long)
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_prompt_ids": prompt,
            "audio_frames": int(attention_mask.sum().item()),
        }


class LengthBucketSampler(torch.utils.data.Sampler):
    """Sortish duration buckets so a batch is 6s-with-7s, not 3s-with-15s.

    Sort by duration, cut into windows of `mix * batch_size`, shuffle inside
    each window, then shuffle the batches. Same idea as fairseq / HF
    LengthGroupedSampler, keyed on clip seconds (not token ids).
    """

    def __init__(self, lengths: list[float], batch_size: int, mix: int = 4, seed: int = 42):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.batch_size = max(int(batch_size), 1)
        self.mix = max(int(mix), 1)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.lengths.size)

    def _batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + epoch)
        order = np.argsort(self.lengths, kind="stable")
        window = max(self.batch_size * self.mix, self.batch_size)
        batches: list[list[int]] = []
        for start in range(0, len(order), window):
            chunk = order[start : start + window].copy()
            rng.shuffle(chunk)
            for i in range(0, len(chunk), self.batch_size):
                batch = chunk[i : i + self.batch_size].tolist()
                if batch:
                    batches.append(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self._batches(self.epoch):
            yield from batch


def duration_pad_waste(durations: list[float], batches: list[list[int]]) -> float:
    wastes = []
    for batch in batches:
        vals = [max(float(durations[i]), 1e-3) for i in batch]
        wastes.append(1.0 - sum(vals) / (len(vals) * max(vals)))
    return float(np.mean(wastes)) if wastes else 0.0


def random_batches(n: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size) if order[i : i + batch_size]]


@dataclass
class CohereCollator:
    processor: object
    pad_token_id: int
    decoder_start_token_id: int
    pad_fracs: list | None = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        prompts = [f["decoder_prompt_ids"] for f in features]
        frames = [int(f.get("audio_frames", att.sum().item())) for f, att in zip(features, attention_mask)]
        max_t = max(feat.size(0) for feat in input_features)
        if self.pad_fracs is not None and max_t > 0:
            self.pad_fracs.append(1.0 - sum(frames) / (len(frames) * max_t))

        padded = _pad_features(input_features, attention_mask)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        decoder_input_ids = shift_tokens_right(
            labels_pad.masked_fill(labels_pad == -100, self.pad_token_id),
            self.pad_token_id,
            self.decoder_start_token_id,
        )
        # Train the transcript, not the language prompt.
        prompt_len = prompts[0].numel()
        labels_pad[:, :prompt_len] = -100
        decoder_prompt_ids = pad_sequence(prompts, batch_first=True, padding_value=self.pad_token_id)
        batch = {
            "input_features": padded["input_features"],
            "attention_mask": padded["attention_mask"],
            "labels": labels_pad,
            "decoder_input_ids": decoder_input_ids,
            "decoder_prompt_ids": decoder_prompt_ids,
        }
        return batch


_COHERE_FORWARD = {
    "input_features",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "decoder_inputs_embeds",
    "decoder_position_ids",
    "encoder_outputs",
    "past_key_values",
    "labels",
    "use_cache",
}


def _model_dtype(model) -> torch.dtype | None:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def _cohere_forward_inputs(inputs: dict, model=None) -> dict:
    # Drop PEFT/Trainer leftovers. CohereAsr decoder is called as
    # decoder(input_ids=decoder_input_ids, **kwargs); a second input_ids
    # (or inputs_embeds) raises "multiple values for keyword argument".
    clean = {k: v for k, v in inputs.items() if k in _COHERE_FORWARD}
    dtype = _model_dtype(model) if model is not None else None
    feats = clean.get("input_features")
    # Train autocast hides this; generate does not. Encoder convs are bf16.
    if dtype is not None and torch.is_tensor(feats) and feats.is_floating_point() and feats.dtype != dtype:
        clean["input_features"] = feats.to(dtype=dtype)
    return clean


class CohereTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        lengths = [item.duration for item in dataset.items]
        return LengthBucketSampler(
            lengths,
            batch_size=self._train_batch_size,
            mix=4,
            seed=self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**_cohere_forward_inputs(dict(inputs), model))
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer
        adapter_params, lora_params, other = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            in_decoder = ".decoder." in f".{name}." or name.startswith("decoder.")
            if "conv_adapter" in name or ("lora_" in name and not in_decoder):
                adapter_params.append(param)
            elif "lora_" in name:
                lora_params.append(param)
            else:
                other.append(param)
        groups = []
        if adapter_params:
            groups.append(
                {"params": adapter_params, "lr": COHERE_TRAIN["encoder_adapter_lr"], "weight_decay": 0.01}
            )
        if lora_params:
            groups.append({"params": lora_params, "lr": COHERE_TRAIN["decoder_lr"], "weight_decay": 0.0})
        if other:
            groups.append(
                {"params": other, "lr": COHERE_TRAIN["decoder_lr"], "weight_decay": COHERE_TRAIN["weight_decay"]}
            )
        if not groups:
            raise RuntimeError("No trainable parameters")
        fused = torch.cuda.is_available()
        try:
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=fused)
        except TypeError:
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
        return self.optimizer

    def _prepare_decoder_input_ids_for_generation(self, batch_size, model_kwargs):
        prompt = model_kwargs.pop("decoder_prompt_ids", None)
        if prompt is not None:
            model_kwargs["decoder_input_ids"] = prompt
        return super()._prepare_decoder_input_ids_for_generation(batch_size, model_kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        inputs = dict(inputs)
        prompt = inputs.pop("decoder_prompt_ids", None)
        clean = _cohere_forward_inputs(inputs, model)
        if prediction_loss_only or not self.args.predict_with_generate:
            return super().prediction_step(
                model, clean, prediction_loss_only, ignore_keys=ignore_keys
            )

        # Prompt is only for generate. Loss needs the full teacher-forced
        # decoder_input_ids (same length as labels). Mixing them is the
        # "batch_size 20 vs 244" crash from ForCausalLMLoss.
        has_labels = "labels" in clean
        clean = self._prepare_inputs(clean)
        if prompt is not None:
            prompt = self._prepare_inputs({"decoder_input_ids": prompt})["decoder_input_ids"]

        if not gen_kwargs and hasattr(self, "_gen_kwargs"):
            gen_kwargs = dict(self._gen_kwargs)
        if gen_kwargs.get("num_beams") is None:
            gen_kwargs.pop("num_beams", None)
        if gen_kwargs.get("max_length") is None:
            gen_kwargs.pop("max_length", None)

        gen_inputs = {
            "input_features": clean["input_features"],
            "attention_mask": clean["attention_mask"],
        }
        if prompt is not None:
            gen_inputs["decoder_input_ids"] = prompt

        with torch.no_grad():
            generated = model.generate(**gen_inputs, **gen_kwargs)
            loss = None
            if has_labels:
                outputs = model(**{k: clean[k] for k in _COHERE_FORWARD if k in clean})
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs.loss).detach().mean()

        labels = clean.get("labels")
        gen_config = getattr(model, "generation_config", None)
        max_len = None
        if gen_config is not None:
            max_len = gen_config.max_length
            if gen_config.max_new_tokens is not None:
                max_len = max(max_len or 0, gen_config.max_new_tokens + 1)
        if max_len is not None:
            if generated.shape[-1] < max_len:
                generated = self._pad_tensors_to_max_len(generated, max_len)
            if labels is not None and labels.shape[-1] < max_len:
                labels = self._pad_tensors_to_max_len(labels, max_len)
        return loss, generated, labels


def gpu_mem_gb() -> dict:
    """Process VRAM (and host RSS). Values are GiB."""
    out = {
        "alloc": 0.0,
        "reserved": 0.0,
        "peak_alloc": 0.0,
        "device_used": 0.0,
        "device_total": 0.0,
        "rss": 0.0,
    }
    if torch.cuda.is_available():
        out["alloc"] = torch.cuda.memory_allocated() / 1024**3
        out["reserved"] = torch.cuda.memory_reserved() / 1024**3
        out["peak_alloc"] = torch.cuda.max_memory_allocated() / 1024**3
        try:
            free, total = torch.cuda.mem_get_info()
            out["device_total"] = total / 1024**3
            out["device_used"] = (total - free) / 1024**3
        except Exception:
            pass
    try:
        import resource

        # Linux ru_maxrss is KiB.
        out["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except Exception:
        pass
    return out


def _fmt_vram(mem: dict, budget: float) -> str:
    used = max(mem["reserved"], mem["device_used"], mem["alloc"])
    return (
        f"vram alloc={mem['alloc']:.2f} reserved={mem['reserved']:.2f} "
        f"peak={mem['peak_alloc']:.2f} device={mem['device_used']:.2f}/{mem['device_total']:.2f} "
        f"budget={used:.2f}/{budget:.1f}GB ({100.0 * used / max(budget, 1e-6):.0f}%) "
        f"host_rss={mem['rss']:.2f}GB"
    )


class CerPrinter(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print({k: metrics[k] for k in metrics if "cer" in k or "wer" in k or "loss" in k})


class LengthBucketCallback(TrainerCallback):
    """Reshuffle buckets each epoch and log real mel-pad waste."""

    def __init__(self, pad_fracs: list):
        self.pad_fracs = pad_fracs
        self._seen = 0

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        sampler = getattr(train_dataloader, "sampler", None) if train_dataloader is not None else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        print(f"[sampler] length_bucket epoch={epoch}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if len(self.pad_fracs) <= self._seen:
            return
        chunk = self.pad_fracs[self._seen :]
        self._seen = len(self.pad_fracs)
        mean_pad = float(np.mean(chunk))
        print(
            f"[pad] step={state.global_step} batch_pad={mean_pad:.1%} "
            f"running={float(np.mean(self.pad_fracs)):.1%} n={len(self.pad_fracs)}",
            flush=True,
        )


class VramMonitor(TrainerCallback):
    """Log GPU memory against the 16GB IFM cap. Eval generate is the spike."""

    def __init__(self, budget_gb: float, log_path: Path | None = None):
        self.budget_gb = float(budget_gb)
        self.log_path = Path(log_path) if log_path else None
        self.history: list[dict] = []
        self.peak = 0.0

    def _record(self, tag: str, state=None) -> dict:
        mem = gpu_mem_gb()
        used = max(mem["reserved"], mem["device_used"], mem["alloc"], mem["peak_alloc"])
        self.peak = max(self.peak, used, mem["peak_alloc"])
        row = {"tag": tag, "step": getattr(state, "global_step", None), **mem, "used": used}
        self.history.append(row)
        print(f"[vram:{tag}] {_fmt_vram(mem, self.budget_gb)}", flush=True)
        if used >= 0.92 * self.budget_gb:
            print(
                f"[vram:WARN] {used:.2f}GB ≥ 92% of the {self.budget_gb:.0f}GB cap. "
                "Drop train batch before raising eval batch.",
                flush=True,
            )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return row

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._record("train_begin_weights_only", state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 2:
            self._record(f"step_{state.global_step}", state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._record("log", state)

    def on_evaluate(self, args, state, control, **kwargs):
        self._record("eval", state)

    def on_train_end(self, args, state, control, **kwargs):
        self._record("train_end", state)
        print(f"[vram:peak] {self.peak:.2f}GB / {self.budget_gb:.1f}GB cap", flush=True)


def _batch_to_device(batch: dict, device: str, model) -> dict:
    dtype = _model_dtype(model)
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point() and dtype is not None:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return _cohere_forward_inputs(moved, model)


def choose_train_batch(
    model,
    dataset,
    collator,
    device: str,
    budget_gb: float,
    start_bs: int = 32,
    step: int = 16,
    max_bs: int = 128,
    target_frac: float = 0.82,
) -> tuple[int, float]:
    """Grow train batch until reserved VRAM hits target_frac of the 16GB cap.

    Grad-checkpoint + LoRA makes activations cheap (~0.1GB / extra clip), so
    the useful way to fill the card is a large micro-batch, not a tiny one.
    Eval generate stays at 1 (~18% headroom). Probe uses the longest clips.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return start_bs, 0.0
    target = budget_gb * target_frac
    ranked = sorted(range(len(dataset)), key=lambda i: dataset.items[i].duration, reverse=True)
    last_ok = start_bs
    last_used = 0.0
    model.train()
    batch_size = start_bs
    while batch_size <= max_bs and batch_size <= len(dataset):
        items = [dataset[i] for i in ranked[:batch_size]]
        try:
            batch = _batch_to_device(collator(items), device, model)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            mem = gpu_mem_gb()
            used = max(mem["reserved"], mem["peak_alloc"], mem["device_used"])
            print(
                f"[vram:probe] train_bs={batch_size} {_fmt_vram(mem, budget_gb)} "
                f"target={target:.2f}GB",
                flush=True,
            )
            model.zero_grad(set_to_none=True)
            del outputs, loss, batch
            torch.cuda.empty_cache()
            if used > target:
                print(
                    f"[vram:probe] bs={batch_size} exceeds {target:.1f}GB target, "
                    f"keeping {last_ok}",
                    flush=True,
                )
                break
            last_ok = batch_size
            last_used = used
            batch_size += step
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[vram:probe] OOM at train_bs={batch_size}, keeping {last_ok}", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    print(f"[vram:probe] chose train_bs={last_ok} used≈{last_used:.2f}GB / {budget_gb:.1f}GB", flush=True)
    return last_ok, last_used


def _dataloader_workers() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(key)
        if raw:
            try:
                return min(4, max(0, int(str(raw).split()[0]) - 1))
            except ValueError:
                pass
    return min(4, max(0, (os.cpu_count() or 2) - 1))


def trainable_summary(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total), "pct": 100.0 * trainable / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=COHERE_MODEL)
    parser.add_argument("--train", type=Path, default=SPLITS_DIR / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=SPLITS_DIR / "dev.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--vram-budget-gb", type=float, default=COHERE_TRAIN["vram_budget_gb"])
    parser.add_argument("--recipe", default="method", choices=sorted(COHERE_RECIPES))
    args = parser.parse_args()
    ensure_dirs()
    recipe = COHERE_RECIPES[args.recipe]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_name = args.output or (CHECKPOINTS_DIR / recipe["output"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_cohere_base(args.model, device, train=True)
    if recipe["conv_adapter"]:
        adapter_meta = prepare_student(model)
    else:
        adapter_meta = {
            "n_layers": 0,
            "start_layer": -1,
            "attached_layers": [],
            "kernels": [],
            "fusion": None,
        }
    model, lora_targets = attach_decoder_lora(model, scope=recipe["lora_scope"])
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    counts = trainable_summary(model)

    pad_id = processor.tokenizer.pad_token_id
    start_id = model.config.decoder_start_token_id
    if start_id is None:
        start_id = processor.get_decoder_prompt_ids(language=COHERE_LANGUAGE, punctuation=True)[0]

    train_ds = CohereJsonlDataset(args.train, processor, augment=True, seed=SEED)
    dev_ds = CohereJsonlDataset(args.dev, processor, augment=False, seed=SEED)
    collator = CohereCollator(processor, pad_token_id=pad_id, decoder_start_token_id=start_id)
    train_bs, _ = choose_train_batch(
        model, train_ds, collator, device, args.vram_budget_gb
    )
    # Keep the update size near 16. Larger micro-batch already fills the card.
    accum = max(1, int(round(16 / train_bs))) if train_bs < 16 else 1
    pad_fracs: list[float] = []
    collator.pad_fracs = pad_fracs
    durations = [item.duration for item in train_ds.items]
    bucket_sampler = LengthBucketSampler(durations, batch_size=train_bs, mix=4, seed=SEED)
    waste_random = duration_pad_waste(durations, random_batches(len(durations), train_bs, SEED))
    waste_bucket = duration_pad_waste(durations, bucket_sampler._batches(0))
    print(
        f"[sampler] length_bucket mix=4 train_bs={train_bs} "
        f"pad_waste_duration random={waste_random:.1%} bucketed={waste_bucket:.1%} "
        f"(1 - sum(dur)/max(dur) per batch)",
        flush=True,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.asarray(pred_ids)
        if pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(axis=-1)
        # Trainer pads gathered preds/labels with -100. tokenizers cannot
        # decode negatives (OverflowError on the u32 cast).
        safe_pad = pad_id if pad_id is not None else (processor.tokenizer.eos_token_id or 0)
        pred_ids = np.where(pred_ids < 0, safe_pad, pred_ids).astype(np.int64)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, safe_pad).astype(np.int64)
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        raw = compute_error_rates(refs, hyps, normalized=False)
        norm = compute_error_rates(refs, hyps, normalized=True)
        return {
            "wer": raw.wer,
            "cer": raw.cer,
            "wer_norm": norm.wer,
            "cer_norm": norm.cer,
        }

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_name),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=COHERE_TRAIN["per_device_eval_batch_size"],
        gradient_accumulation_steps=accum,
        learning_rate=COHERE_TRAIN["decoder_lr"],
        # transformers 5: float in [0, 1) is a warmup ratio (warmup_ratio is gone).
        warmup_steps=COHERE_TRAIN["warmup_ratio"],
        lr_scheduler_type="cosine",
        num_train_epochs=COHERE_TRAIN["num_train_epochs"],
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=COHERE_TRAIN["eval_steps"],
        save_strategy="steps",
        save_steps=COHERE_TRAIN["eval_steps"],
        logging_steps=COHERE_TRAIN["logging_steps"],
        predict_with_generate=True,
        generation_max_length=COHERE_TRAIN["max_new_tokens"],
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        fp16_full_eval=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_grad_norm=COHERE_TRAIN["max_grad_norm"],
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="cer_norm",
        greater_is_better=False,
        save_total_limit=COHERE_TRAIN["save_total_limit"],
        seed=SEED,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=_dataloader_workers(),
    )

    vram_cb = VramMonitor(args.vram_budget_gb, log_path=run_name / "vram.jsonl")
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[CerPrinter(), LengthBucketCallback(pad_fracs), vram_cb],
    )
    try:
        trainer = CohereTrainer(processing_class=processor, **trainer_kwargs)
    except TypeError:
        trainer = CohereTrainer(tokenizer=processor.tokenizer, **trainer_kwargs)

    epochs = COHERE_TRAIN["num_train_epochs"]
    steps_per_epoch = max(1, -(-len(train_ds) // (train_bs * accum)))
    planned_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * epochs
    encoder_bit = (
        f"encoder_multiconv=layers{adapter_meta['start_layer']}-{adapter_meta['n_layers'] - 1} "
        f"k={adapter_meta['kernels']} fusion={adapter_meta['fusion']}"
        if recipe["conv_adapter"]
        else f"encoder={recipe['lora_scope']}_lora" if recipe["lora_scope"] != "decoder" else "encoder=frozen"
    )
    print(
        f"Training {args.model} recipe={args.recipe} "
        f"lora_scope={recipe['lora_scope']} r={COHERE_LORA['r']} "
        f"{encoder_bit} "
        f"trainable={counts['trainable']} ({counts['pct']:.3f}%) "
        f"n_train={len(train_ds)} n_dev={len(dev_ds)} "
        f"epochs={epochs} steps/epoch≈{steps_per_epoch} planned_steps≈{planned_steps} "
        f"train_bs={train_bs} eval_bs={COHERE_TRAIN['per_device_eval_batch_size']} "
        f"accum={accum} effective_bs={train_bs * accum} "
        f"sampler=length_bucket mix=4 "
        f"pad_waste_random={waste_random:.1%} pad_waste_bucket={waste_bucket:.1%} "
        f"vram_budget={args.vram_budget_gb:.0f}GB "
        f"precision={'bf16' if use_bf16 else 'fp16' if torch.cuda.is_available() else 'fp32'} "
        f"wave_aug={getattr(train_ds.wave_aug, 'backend', None)} "
        f"musan={COHERE_TRAIN['musan_prob']:.0%} rir={COHERE_TRAIN['rir_prob']:.0%}"
    )
    trainer.train()

    best = run_name / "best"
    meta = {
        "base_model": args.model,
        "recipe": args.recipe,
        "lora_scope": recipe["lora_scope"],
        "conv_adapter": recipe["conv_adapter"],
        "language": COHERE_LANGUAGE,
        "lora": dict(COHERE_LORA),
        "lora_targets": lora_targets,
        "skip_bottom_frac": COHERE_ADAPTER["skip_bottom_frac"],
        **adapter_meta,
        "trainable": counts,
        "decoder_lr": COHERE_TRAIN["decoder_lr"],
        "encoder_adapter_lr": COHERE_TRAIN["encoder_adapter_lr"],
        "optimizer": "adamw",
        "betas": [0.9, 0.98],
        "scheduler": "cosine+warmup_ratio_0.1",
        "precision": "bf16" if use_bf16 else "fp16",
        "seed": SEED,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "vram_budget_gb": args.vram_budget_gb,
        "vram_peak_gb": vram_cb.peak,
        "sampler": "length_bucket",
        "sampler_mix": 4,
        "pad_waste_duration_random": waste_random,
        "pad_waste_duration_bucket": waste_bucket,
        "pad_waste_mel_running": float(np.mean(pad_fracs)) if pad_fracs else None,
    }
    save_student(trainer.model, processor, best, meta)
    if recipe["conv_adapter"]:
        enc_line = "encoder=frozen Conformer + MultiConvAdapter (layers 16-47, bot=64, k=7,15,23,31)\n"
    elif recipe["lora_scope"] == "decoder":
        enc_line = "encoder=frozen\n"
    else:
        enc_line = "encoder=LoRA qkvo\n"
    dec_line = (
        "decoder=frozen\n"
        if recipe["lora_scope"] == "encoder"
        else "decoder=LoRA r=32 a=64 qkvo self-attn+cross-attn (not MLP, not embed)\n"
    )
    (run_name / "DECISIONS.txt").write_text(
        f"recipe={args.recipe} conv_adapter={recipe['conv_adapter']} lora_scope={recipe['lora_scope']}\n"
        "model=CohereLabs/cohere-transcribe-arabic-07-2026\n"
        f"{enc_line}{dec_line}"
        "lr=1e-4 encoder adapter and decoder LoRA\n"
        "opt=AdamW betas=(0.9, 0.98) cosine 10% warmup\n"
        "precision=bf16 (fp16 fallback) + grad checkpoint + tf32\n"
        "aug=speed{0.9,1.0,1.1} gain±3dB drop-chunk SpecAugment + MUSAN 25% + RIR 25%\n"
        "prompt_masked=true language=ar code_switch_kept=true\n"
        "sampler=length_bucket mix=4 (sortish duration windows; eval stays batch 1)\n"
        f"pad_waste_duration random={waste_random:.3f} bucketed={waste_bucket:.3f}\n"
        f"single_seed={SEED}\n"
    )
    write_json(run_name / "run_meta.json", meta)
    print(f"Saved {best}")


if __name__ == "__main__":
    main()
