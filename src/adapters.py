"""Residual MultiConvAdapter on the frozen Cohere Conformer.

Kernels K={7,15,23,31} + concat_fusion from MULTI-CONVFORMER
(Prabhu et al. 2024). Skip the bottom third of layers. Zero-init the
up-projection so the first step is identity.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

DEFAULT_KERNELS = (7, 15, 23, 31)
DEFAULT_FUSION = "concat_fusion"
DEFAULT_MERGE_KERNEL = 31


class MultiConvAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        bottleneck: int = 64,
        kernels: Iterable[int] = DEFAULT_KERNELS,
        dropout: float = 0.1,
        fusion: str = DEFAULT_FUSION,
        merge_kernel: int = DEFAULT_MERGE_KERNEL,
    ):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels:
            raise ValueError("Need at least one convolution kernel")
        if any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError(f"Kernels must be odd and positive, got {kernels}")
        if bottleneck < len(kernels) or bottleneck % 2 != 0:
            raise ValueError(f"bottleneck must be even and >= n_kernels, got {bottleneck}")
        if fusion not in {"sum", "weighted_sum", "concat", "concat_fusion"}:
            raise ValueError(f"Unknown fusion={fusion}")
        if fusion in {"concat", "concat_fusion"} and bottleneck % len(kernels) != 0:
            raise ValueError(
                f"concat fusion needs bottleneck ({bottleneck}) divisible by "
                f"{len(kernels)} kernels"
            )

        self.kernels = kernels
        self.fusion = fusion
        self.merge_kernel = int(merge_kernel)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, 2 * bottleneck)
        self.act = nn.GELU()
        self.gate_norm = nn.LayerNorm(bottleneck)
        n_kernels = len(kernels)

        if fusion in {"sum", "weighted_sum"}:
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        bottleneck,
                        bottleneck,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                        groups=bottleneck,
                    )
                    for k in kernels
                ]
            )
        else:
            per = bottleneck // n_kernels
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        bottleneck,
                        per,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                        groups=per,
                    )
                    for k in kernels
                ]
            )

        if fusion == "weighted_sum":
            self.kernel_mix = nn.Sequential(
                nn.Linear(bottleneck * n_kernels, n_kernels),
                nn.Softmax(dim=-1),
            )
        else:
            self.kernel_mix = None

        if fusion == "concat_fusion":
            self.merge = nn.Conv1d(
                bottleneck,
                bottleneck,
                kernel_size=self.merge_kernel,
                padding=(self.merge_kernel - 1) // 2,
                groups=bottleneck,
            )
        else:
            self.merge = None

        self.up = nn.Linear(bottleneck, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _fuse(self, branches: list[torch.Tensor]) -> torch.Tensor:
        if self.fusion in {"sum", "weighted_sum"}:
            stacked = torch.stack(branches, dim=-2)
            if self.kernel_mix is not None:
                weights = self.kernel_mix(torch.cat(branches, dim=-1))
                stacked = weights.unsqueeze(-1) * stacked
            return stacked.sum(dim=-2)
        fused = torch.cat(branches, dim=-1)
        if self.merge is not None:
            extra = self.merge(fused.transpose(1, 2)).transpose(1, 2)
            fused = fused + extra
        return fused

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, time, dim)
        residual = hidden_states
        hidden = self.act(self.down(self.norm(hidden_states)))
        left, right = hidden.chunk(2, dim=-1)
        right = self.gate_norm(right).transpose(1, 2)
        branches = [conv(right).transpose(1, 2) for conv in self.convs]
        hidden = left * self._fuse(branches)
        return residual + self.drop(self.up(hidden))


class EncoderBlockWithConvAdapter(nn.Module):
    def __init__(self, block: nn.Module, adapter: MultiConvAdapter):
        super().__init__()
        self.block = block
        self.conv_adapter = adapter

    def forward(self, *args, **kwargs):
        hidden_states = self.block(*args, **kwargs)
        if isinstance(hidden_states, tuple):
            return (self.conv_adapter(hidden_states[0]),) + hidden_states[1:]
        return self.conv_adapter(hidden_states)


def get_encoder(model: nn.Module) -> nn.Module:
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(core, "model") and hasattr(core.model, "encoder"):
        return core.model.encoder
    if hasattr(core, "encoder"):
        return core.encoder
    raise AttributeError("Could not find a Conformer / Parakeet encoder on this model")


def attach_multiconv_adapters(
    model: nn.Module,
    *,
    bottleneck: int,
    kernels: Iterable[int],
    dropout: float,
    skip_bottom_frac: float,
    fusion: str = DEFAULT_FUSION,
    merge_kernel: int = DEFAULT_MERGE_KERNEL,
) -> dict:
    encoder = get_encoder(model)
    layers = encoder.layers
    n_layers = len(layers)
    start = int(n_layers * skip_bottom_frac)
    d_model = int(getattr(encoder.config, "hidden_size", 1280))
    attached = []
    for idx in range(start, n_layers):
        block = layers[idx]
        if isinstance(block, EncoderBlockWithConvAdapter):
            continue
        adapter = MultiConvAdapter(
            d_model,
            bottleneck=bottleneck,
            kernels=kernels,
            dropout=dropout,
            fusion=fusion,
            merge_kernel=merge_kernel,
        )
        try:
            ref = next(block.parameters())
            adapter.to(device=ref.device, dtype=ref.dtype)
        except StopIteration:
            pass
        layers[idx] = EncoderBlockWithConvAdapter(block, adapter)
        attached.append(idx)
    return {
        "n_layers": n_layers,
        "start_layer": start,
        "attached_layers": attached,
        "d_model": d_model,
        "bottleneck": bottleneck,
        "kernels": list(kernels),
        "fusion": fusion,
        "merge_kernel": merge_kernel,
        "dropout": dropout,
    }


def _in_decoder(name: str) -> bool:
    dotted = f".{name}."
    return ".decoder." in dotted or name.startswith("decoder.")


def decoder_lora_targets(model: nn.Module, kinds: Iterable[str]) -> list[str]:
    """Only the 8-layer decoder (self-attn + cross-attn). Never the encoder."""
    kinds = tuple(kinds)
    names = []
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in kinds:
            continue
        if not _in_decoder(name):
            continue
        if "self_attn" not in name and "encoder_attn" not in name:
            continue
        names.append(name)
    if not names:
        raise RuntimeError(
            "No decoder attention projections found. "
            "Is this CohereAsrForConditionalGeneration?"
        )
    return names


def encoder_lora_targets(model: nn.Module, kinds: Iterable[str]) -> list[str]:
    """Conformer self-attn q/k/v/o. Never decoder, never MLP."""
    kinds = tuple(kinds)
    names = []
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in kinds:
            continue
        if _in_decoder(name):
            continue
        if "self_attn" not in name:
            continue
        names.append(name)
    if not names:
        raise RuntimeError("No encoder attention projections found.")
    return names


def lora_targets_for_scope(model: nn.Module, scope: str, kinds: Iterable[str]) -> list[str]:
    if scope == "decoder":
        return decoder_lora_targets(model, kinds)
    if scope == "encoder":
        return encoder_lora_targets(model, kinds)
    if scope == "full":
        return decoder_lora_targets(model, kinds) + encoder_lora_targets(model, kinds)
    raise ValueError(f"Unknown lora scope={scope}")
