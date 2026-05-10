from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension >= 2, got dim={dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self._cos_sin(x.size(-2), device=x.device, dtype=x.dtype)
        return (x * cos) + (_rotate_half(x) * sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary(q), self.apply_rotary(k)
