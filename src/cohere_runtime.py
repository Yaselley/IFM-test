"""Load / save / decode Cohere Transcribe Arabic students."""

from __future__ import annotations

from pathlib import Path

import torch

from src.adapters import attach_multiconv_adapters, lora_targets_for_scope
from src.config import COHERE_ADAPTER, COHERE_LANGUAGE, COHERE_LORA, COHERE_MODEL, SAMPLE_RATE
from src.io_utils import hub_kw, load_mono_16k, write_json


def load_cohere_base(repo: str, device: str, *, train: bool = False):
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(repo, **hub_kw())
    model = CohereAsrForConditionalGeneration.from_pretrained(
        repo, dtype=torch.bfloat16, **hub_kw()
    )
    model.config.use_cache = not train
    if train:
        model.gradient_checkpointing_enable()
    model.to(device)
    return model, processor


def attach_decoder_lora(model, kinds=COHERE_LORA["target_kinds"], scope: str = "decoder"):
    from peft import LoraConfig, get_peft_model

    targets = lora_targets_for_scope(model, scope, kinds)
    # No TaskType.SEQ_2_SEQ_LM: that wrapper always calls the base model with
    # input_ids=..., and CohereAsr then does decoder(input_ids=decoder_input_ids,
    # **kwargs) — so input_ids is passed twice.
    has_conv = any(isinstance(getattr(m, "conv_adapter", None), torch.nn.Module) for m in model.modules())
    lora_kwargs = dict(
        r=COHERE_LORA["r"],
        lora_alpha=COHERE_LORA["lora_alpha"],
        lora_dropout=COHERE_LORA["lora_dropout"],
        bias=COHERE_LORA["bias"],
        target_modules=targets,
    )
    if has_conv:
        lora_kwargs["modules_to_save"] = ["conv_adapter"]
    config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, config)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # PEFT clones conv_adapter onto CPU. Only move real modules, not the
    # config set of names also stored as `modules_to_save`.
    for module in model.modules():
        adapter = getattr(module, "conv_adapter", None)
        if isinstance(adapter, torch.nn.Module):
            adapter.to(device=device, dtype=dtype)
        saved = getattr(module, "modules_to_save", None)
        if isinstance(saved, torch.nn.Module):
            saved.to(device=device, dtype=dtype)
    for name, param in model.named_parameters():
        if "conv_adapter" in name:
            param.requires_grad = True
            if param.device != device:
                param.data = param.data.to(device=device, dtype=dtype)
    return model, targets


def prepare_student(model, adapter_cfg: dict | None = None):
    cfg = dict(COHERE_ADAPTER)
    if adapter_cfg:
        cfg.update(adapter_cfg)
    meta = attach_multiconv_adapters(
        model,
        bottleneck=cfg["bottleneck"],
        kernels=cfg["kernels"],
        dropout=cfg["dropout"],
        skip_bottom_frac=cfg["skip_bottom_frac"],
        fusion=cfg.get("fusion", "concat_fusion"),
        merge_kernel=cfg.get("merge_kernel", 31),
    )
    return meta


def _encoder_adapter_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "conv_adapter" in k}


def save_student(model, processor, output: Path, meta: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    write_json(output / "adapter_meta.json", meta)
    torch.save(_encoder_adapter_state(model), output / "encoder_adapters.pt")


def load_cohere_student(path: str, device: str):
    path = Path(path)
    meta_path = path / "adapter_meta.json"
    if not meta_path.exists():
        model, processor = load_cohere_base(str(path), device, train=False)
        model.eval()
        return model, processor
    meta = __import__("json").loads(meta_path.read_text())
    base_id = meta.get("base_model") or COHERE_MODEL
    model, processor = load_cohere_base(base_id, device, train=False)
    if meta.get("attached_layers"):
        prepare_student(
            model,
            {
                "bottleneck": meta["bottleneck"],
                "kernels": meta["kernels"],
                "dropout": meta["dropout"],
                "skip_bottom_frac": meta.get("skip_bottom_frac", 0.33),
                "fusion": meta.get("fusion", "concat_fusion"),
                "merge_kernel": meta.get("merge_kernel", 31),
            },
        )
    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(path))
    extra = path / "encoder_adapters.pt"
    if extra.exists():
        missing = model.load_state_dict(torch.load(extra, map_location="cpu"), strict=False)
        _ = missing
    model.to(device).eval()
    return model, processor


@torch.inference_mode()
def transcribe_cohere(model, processor, wav: torch.Tensor, device: str, language: str = COHERE_LANGUAGE) -> str:
    dtype = getattr(model, "dtype", None) or torch.bfloat16
    inputs = processor(
        wav.detach().cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        language=language,
    )
    if hasattr(inputs, "to"):
        try:
            inputs = inputs.to(device, dtype=dtype)
        except TypeError:
            inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=128)
    chunk = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
    try:
        text = processor.decode(
            out, skip_special_tokens=True, audio_chunk_index=chunk, language=language
        )
        if isinstance(text, (list, tuple)):
            text = text[0]
    except TypeError:
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return str(text).strip()


def transcribe_cohere_path(model, processor, path: str, device: str, language: str = COHERE_LANGUAGE) -> str:
    wav = load_mono_16k(path, SAMPLE_RATE)
    return transcribe_cohere(model, processor, wav, device, language=language)
