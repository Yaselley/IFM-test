"""Speed / gain / drop-chunk / SpecAugment, plus MUSAN and RIR at 25%."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio

AUDIO_EXTS = {".wav", ".flac", ".ogg"}


def _try_speechbrain():
    try:
        from speechbrain.augment.time_domain import SpeedPerturb, DropChunk

        return SpeedPerturb, DropChunk
    except Exception:
        return None, None


def _iter_audio(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    out: list[str] = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTS:
                out.append(str(Path(dirpath) / name))
    out.sort()
    return out


def cached_file_list(root: Path, cache: Path, *, exclude_substr: tuple[str, ...] = ()) -> list[str]:
    if cache.is_file():
        return [line for line in cache.read_text().splitlines() if line]
    paths = _iter_audio(root)
    if exclude_substr:
        tokens = tuple(s.lower() for s in exclude_substr)
        paths = [p for p in paths if not any(tok in Path(p).name.lower() for tok in tokens)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(paths) + ("\n" if paths else ""))
    return paths


def load_mono(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if int(sr) != int(sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
    return wav


def mix_at_snr(wav: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    n = int(wav.numel())
    if n < 16 or noise.numel() < 16:
        return wav
    if noise.numel() < n:
        reps = (n + noise.numel() - 1) // noise.numel()
        noise = noise.repeat(reps)
    extra = noise.numel() - n
    start = int(torch.randint(0, extra + 1, (1,)).item()) if extra > 0 else 0
    noise = noise[start : start + n]
    power = torch.mean(wav * wav).clamp_min(1e-8)
    noise_power = torch.mean(noise * noise).clamp_min(1e-8)
    scale = torch.sqrt(power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return wav + scale * noise


def convolve_rir(wav: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    if rir.numel() < 8:
        return wav
    rir = rir / rir.abs().max().clamp_min(1e-8)
    mixed = torchaudio.functional.fftconvolve(wav, rir, mode="full")
    peak = int(rir.abs().argmax().item())
    out = mixed[peak : peak + wav.numel()]
    if out.numel() < wav.numel():
        out = torch.nn.functional.pad(out, (0, wav.numel() - out.numel()))
    src_rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
    out_rms = out.pow(2).mean().sqrt().clamp_min(1e-8)
    return (out * (src_rms / out_rms)).to(dtype=wav.dtype)


class WaveformAugment:
    def __init__(
        self,
        sample_rate: int,
        speed_factors: Iterable[float],
        gain_db: float,
        drop_chunk_prob: float,
        drop_chunk_max_frac: float,
        noise_prob: float,
        noise_snr: tuple[float, float],
        seed: int,
        rir_prob: float = 0.25,
        musan_prob: float | None = None,
        musan_root: str | Path | None = None,
        rir_root: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.sample_rate = sample_rate
        self.speed_factors = tuple(float(x) for x in speed_factors)
        self.gain_db = float(gain_db)
        self.drop_chunk_prob = float(drop_chunk_prob)
        self.drop_chunk_max_frac = float(drop_chunk_max_frac)
        self.noise_prob = float(musan_prob if musan_prob is not None else noise_prob)
        self.rir_prob = float(rir_prob)
        self.noise_snr = (float(noise_snr[0]), float(noise_snr[1]))
        self.rng = np.random.default_rng(seed)

        SpeedPerturb, DropChunk = _try_speechbrain()
        self.sb_speed = None
        self.sb_drop = None
        if SpeedPerturb is not None:
            percents = [int(round(f * 100)) for f in self.speed_factors]
            try:
                self.sb_speed = SpeedPerturb(orig_freq=sample_rate, speeds=percents)
            except TypeError:
                self.sb_speed = SpeedPerturb(orig_freq=sample_rate, speeds=percents, perturb_prob=1.0)
        if DropChunk is not None:
            try:
                self.sb_drop = DropChunk()
            except TypeError:
                self.sb_drop = None
        self.backend = "speechbrain" if self.sb_speed is not None else "torch"

        cache = Path(cache_dir) if cache_dir else Path("data/cache")
        self.musan_noise: list[str] = []
        self.musan_music: list[str] = []
        self.rirs: list[str] = []
        if musan_root:
            root = Path(musan_root)
            self.musan_noise = cached_file_list(root / "noise", cache / "musan_noise.txt")
            self.musan_music = cached_file_list(root / "music", cache / "musan_music.txt")
        if rir_root:
            root = Path(rir_root)
            simulated = cached_file_list(root / "simulated_rirs", cache / "rir_simulated.txt")
            real = cached_file_list(
                root / "real_rirs_isotropic_noises",
                cache / "rir_real.txt",
                exclude_substr=("noise",),
            )
            self.rirs = simulated + real
        extras = []
        if self.musan_noise or self.musan_music:
            extras.append(f"musan{len(self.musan_noise)}+music{len(self.musan_music)}")
        if self.rirs:
            extras.append(f"rir{len(self.rirs)}")
        if extras:
            self.backend = f"{self.backend}+{'+'.join(extras)}"

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.float()
        wav = self._speed(wav)
        wav = self._gain(wav)
        if self.rirs and self.rng.random() < self.rir_prob:
            wav = self._rir(wav)
        if self.rng.random() < self.noise_prob:
            wav = self._noise(wav)
        if self.rng.random() < self.drop_chunk_prob:
            wav = self._drop_chunk(wav)
        return wav

    def _speed(self, wav: torch.Tensor) -> torch.Tensor:
        if self.sb_speed is not None:
            out = self.sb_speed(wav.unsqueeze(0))
            return out.squeeze(0) if torch.is_tensor(out) else wav
        factor = float(self.rng.choice(self.speed_factors))
        if abs(factor - 1.0) < 1e-6:
            return wav
        # rate * factor snapped to 100Hz, so the ratio stays rational. Dividing
        # (16000 / 0.9 = 17778) leaves gcd 2 and torchaudio builds an
        # 8889 x 96001 resampling kernel, several GB for one clip.
        target_sr = max(int(round(self.sample_rate * factor / 100.0)) * 100, 1000)
        return torchaudio.functional.resample(wav, self.sample_rate, target_sr)

    def _gain(self, wav: torch.Tensor) -> torch.Tensor:
        db = float(self.rng.uniform(-self.gain_db, self.gain_db))
        return wav * (10.0 ** (db / 20.0))

    def _drop_chunk(self, wav: torch.Tensor) -> torch.Tensor:
        n = int(wav.numel())
        if n < 400:
            return wav
        if self.sb_drop is not None:
            try:
                lengths = torch.tensor([n])
                out = self.sb_drop(wav.unsqueeze(0), lengths)
                return out.squeeze(0)
            except Exception:
                pass
        width = max(1, int(self.rng.uniform(0.02, self.drop_chunk_max_frac) * n))
        start = int(self.rng.integers(0, max(n - width, 1)))
        out = wav.clone()
        out[start : start + width] = 0
        return out

    def _pick(self, paths: list[str]) -> str | None:
        if not paths:
            return None
        return paths[int(self.rng.integers(0, len(paths)))]

    def _noise(self, wav: torch.Tensor) -> torch.Tensor:
        snr = float(self.rng.uniform(*self.noise_snr))
        pool = self.musan_noise
        if self.musan_music and self.rng.random() < 0.2:
            pool = self.musan_music
        path = self._pick(pool)
        if path is None:
            return mix_at_snr(wav, torch.randn_like(wav), snr)
        try:
            noise = load_mono(path, self.sample_rate)
        except Exception:
            noise = torch.randn_like(wav)
        return mix_at_snr(wav, noise, snr)

    def _rir(self, wav: torch.Tensor) -> torch.Tensor:
        path = self._pick(self.rirs)
        if path is None:
            return wav
        try:
            rir = load_mono(path, self.sample_rate)
        except Exception:
            return wav
        return convolve_rir(wav, rir)


def spec_augment_mel(
    features: torch.Tensor,
    *,
    n_time: int,
    n_freq: int,
    time_frac: float,
    freq_frac: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """features: (time, n_mels) or (n_mels, time) — we take last dim as freq if ≤128."""
    if features.ndim != 2:
        return features
    n0, n1 = features.shape
    time_axis, freq_axis = (0, 1) if n1 <= n0 else (1, 0)
    n_frames = features.shape[time_axis]
    n_mels = features.shape[freq_axis]
    out = features.clone()
    for _ in range(n_time):
        if n_frames < 8:
            break
        width = max(1, int(time_frac * n_frames))
        start = int(rng.integers(0, max(n_frames - width, 1)))
        if time_axis == 0:
            out[start : start + width, :] = 0
        else:
            out[:, start : start + width] = 0
    for _ in range(n_freq):
        if n_mels < 8:
            break
        width = max(1, int(freq_frac * n_mels))
        start = int(rng.integers(0, max(n_mels - width, 1)))
        if freq_axis == 1:
            out[:, start : start + width] = 0
        else:
            out[start : start + width, :] = 0
    return out
