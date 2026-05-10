from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_perm_mixed_ablation import distribution_loss


def _bool_env_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _require_btd(probs: torch.Tensor, *, name: str = "probs") -> None:
    if probs.ndim != 3:
        raise ValueError(f"expected {name} with shape (B,T,D), got {tuple(probs.shape)}")


def _iter_context_window_chunks(
    probs: torch.Tensor,
    *,
    context_len: int,
    window_batch_size: int,
):
    _require_btd(probs)
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    if probs.size(1) <= 0:
        raise ValueError("sequence length must be positive")

    batch_size, total_steps, prob_dim = probs.shape
    left_pad = probs[:, :1, :].expand(-1, context_len - 1, -1) if context_len > 1 else probs[:, :0, :]
    padded = torch.cat([left_pad, probs], dim=1)
    # `unfold` creates a view, so we only materialize per-chunk windows below.
    all_windows = padded.transpose(1, 2).unfold(dimension=2, size=context_len, step=1)
    steps_per_chunk = max(1, window_batch_size // max(1, batch_size))

    for start in range(0, total_steps, steps_per_chunk):
        end = min(total_steps, start + steps_per_chunk)
        chunk = all_windows[:, :, start:end, :]
        chunk = chunk.permute(0, 2, 3, 1).contiguous().view(-1, context_len, prob_dim)
        yield start, end, chunk


def _temporal_standardize(probs: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return probs
    means = probs.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(torch.var(probs, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    return (probs - means) / stdev.clamp_min(1e-5)


class MovingAverage1d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.pad_left = (self.kernel_size - 1) // 2
        self.pad_right = self.kernel_size - 1 - self.pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        padded = F.pad(x_t, (self.pad_left, self.pad_right), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        return trend.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage1d(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class InvertedEmbedding(nn.Module):
    def __init__(self, *, prob_dim: int, max_seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(max_seq_len, d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, prob_dim, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_embedding(x.transpose(1, 2))
        tokens = tokens + self.channel_embedding[:, : tokens.size(1), :]
        return self.dropout(tokens)


class SequenceEmbedding(nn.Module):
    def __init__(
        self,
        *,
        prob_dim: int,
        max_seq_len: int,
        d_model: int,
        dropout: float,
        use_position: bool = True,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.value_embedding = nn.Linear(prob_dim, d_model)
        self.position_embedding = (
            nn.Parameter(torch.zeros(1, self.max_seq_len, d_model)) if use_position else None
        )
        self.dropout = nn.Dropout(dropout)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"expected at most {self.max_seq_len} time steps, got {x.size(1)}")
        tokens = self.value_embedding(x)
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding[:, : x.size(1), :]
        return self.dropout(tokens)


class ProbSparseAttention(nn.Module):
    def __init__(self, *, mask_flag: bool, factor: int, attention_dropout: float):
        super().__init__()
        self.mask_flag = bool(mask_flag)
        self.factor = int(factor)
        self.dropout = nn.Dropout(attention_dropout)

    def _sample_qk(self, q: torch.Tensor, k: torch.Tensor, *, sample_k: int, n_top: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_heads, q_len, head_dim = q.shape
        k_len = k.size(2)
        index_sample = torch.randint(k_len, (q_len, sample_k), device=q.device)
        k_expand = k.unsqueeze(2).expand(batch_size, n_heads, q_len, k_len, head_dim)
        q_index = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_sample = k_expand[:, :, q_index, index_sample, :]
        qk_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)
        sparsity = qk_sample.max(dim=-1).values - qk_sample.mean(dim=-1)
        top_index = sparsity.topk(n_top, dim=-1, sorted=False).indices
        q_reduce = q.gather(2, top_index.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        qk = torch.matmul(q_reduce, k.transpose(-2, -1))
        return qk, top_index

    def _initial_context(self, v: torch.Tensor, *, q_len: int) -> torch.Tensor:
        if not self.mask_flag:
            context = v.mean(dim=2, keepdim=True)
            return context.expand(-1, -1, q_len, -1).clone()
        if q_len != v.size(2):
            raise ValueError("causal ProbSparse attention requires query and value lengths to match")
        return v.cumsum(dim=2)

    def _update_context(
        self,
        context: torch.Tensor,
        v: torch.Tensor,
        scores: torch.Tensor,
        top_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_heads, _, v_dim = v.shape
        if self.mask_flag:
            key_positions = torch.arange(v.size(2), device=v.device).view(1, 1, 1, -1)
            causal_mask = key_positions > top_index.unsqueeze(-1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        updates = torch.matmul(attn, v)
        updates = updates.to(dtype=context.dtype)
        batch_index = torch.arange(batch_size, device=v.device)[:, None, None]
        head_index = torch.arange(n_heads, device=v.device)[None, :, None]
        context[batch_index, head_index, top_index, :] = updates
        return context

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        del attn_mask
        batch_size, q_len, n_heads, head_dim = queries.shape
        _, k_len, _, _ = keys.shape

        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        sample_k = min(k_len, max(1, int(self.factor * math.ceil(math.log(max(k_len, 2))))))
        n_top = min(q_len, max(1, int(self.factor * math.ceil(math.log(max(q_len, 2))))))
        scores, top_index = self._sample_qk(q, k, sample_k=sample_k, n_top=n_top)
        scores = scores * (1.0 / math.sqrt(head_dim))
        context = self._initial_context(v, q_len=q_len)
        context = self._update_context(context, v, scores, top_index)
        return context.transpose(1, 2).contiguous(), None


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, inner_attention: nn.Module, *, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        self.n_heads = int(n_heads)
        self.query_projection = nn.Linear(d_model, head_dim * self.n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * self.n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * self.n_heads)
        self.out_projection = nn.Linear(head_dim * self.n_heads, d_model)
        self.inner_attention = inner_attention

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = queries.shape
        k_len = keys.size(1)
        queries = self.query_projection(queries).view(batch_size, q_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, k_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, k_len, self.n_heads, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        out = out.view(batch_size, q_len, -1)
        return self.out_projection(out), attn


class InformerConvLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    def __init__(self, attention: MultiHeadAttentionLayer, *, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(attn_out)
        y = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attn


class InformerEncoder(nn.Module):
    def __init__(self, layers: list[InformerEncoderLayer], *, conv_layers: Optional[list[InformerConvLayer]] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = nn.LayerNorm(layers[-1].norm2.normalized_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        attns = []
        if self.conv_layers is not None:
            for layer, conv in zip(self.layers[:-1], self.conv_layers):
                x, attn = layer(x)
                x = conv(x)
                attns.append(attn)
            x, attn = self.layers[-1](x)
            attns.append(attn)
        else:
            for layer in self.layers:
                x, attn = layer(x)
                attns.append(attn)
        return self.norm(x), attns


class InformerDecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionLayer,
        cross_attention: MultiHeadAttentionLayer,
        *,
        d_model: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, memory, memory, attn_mask=None)[0])
        y = self.norm2(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm3(x + y)


class InformerDecoder(nn.Module):
    def __init__(self, layers: list[InformerDecoderLayer], *, d_model: int, prob_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, prob_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory)
        x = self.norm(x)
        return self.projection(x)


class AutoCorrelationAttention(nn.Module):
    def __init__(self, *, factor: int, attention_dropout: float):
        super().__init__()
        self.factor = int(factor)
        self.dropout = nn.Dropout(attention_dropout)

    def _time_delay_aggregate(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, n_channels, length = values.shape
        top_k = min(length, max(1, int(self.factor * math.log(max(length, 2)))))
        mean_corr = corr.mean(dim=1).mean(dim=1)
        if self.training:
            delay_index = torch.topk(mean_corr.mean(dim=0), top_k, dim=-1).indices
            weights = torch.stack([mean_corr[:, delay_index[idx]] for idx in range(top_k)], dim=-1)
            attn = torch.softmax(weights, dim=-1)
            agg = torch.zeros_like(values)
            for idx in range(top_k):
                shifted = torch.roll(values, shifts=-int(delay_index[idx]), dims=-1)
                agg = agg + shifted * attn[:, idx].view(batch_size, 1, 1, 1)
            return agg

        weights, delay = torch.topk(mean_corr, top_k, dim=-1)
        attn = torch.softmax(weights, dim=-1)
        values_twice = values.repeat(1, 1, 1, 2)
        base_index = torch.arange(length, device=values.device).view(1, 1, 1, length)
        agg = torch.zeros_like(values)
        for idx in range(top_k):
            gather_index = base_index + delay[:, idx].view(batch_size, 1, 1, 1)
            shifted = torch.gather(values_twice, dim=-1, index=gather_index.expand(-1, n_heads, n_channels, -1))
            agg = agg + shifted * attn[:, idx].view(batch_size, 1, 1, 1)
        return agg

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        del attn_mask
        out_dtype = queries.dtype
        _, q_len, _, _ = queries.shape
        _, k_len, _, _ = values.shape
        target_len = max(q_len, k_len)

        if q_len < target_len:
            q_pad = queries.new_zeros(queries.size(0), target_len - q_len, queries.size(2), queries.size(3))
            queries = torch.cat([queries, q_pad], dim=1)
        elif q_len > target_len:
            queries = queries[:, :target_len, :, :]

        if k_len < target_len:
            kv_pad = values.new_zeros(values.size(0), target_len - k_len, values.size(2), values.size(3))
            values = torch.cat([values, kv_pad], dim=1)
            keys = torch.cat([keys, kv_pad], dim=1)
        else:
            values = values[:, :target_len, :, :]
            keys = keys[:, :target_len, :, :]

        # FFT kernels do not support bfloat16 on the cluster runtime, so do the
        # frequency-domain correlation in float32 and cast the final output back.
        q_fft = torch.fft.rfft(queries.float().permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.float().permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=target_len, dim=-1)
        aggregated = self._time_delay_aggregate(values.float().permute(0, 2, 3, 1).contiguous(), corr)
        out = aggregated.permute(0, 3, 1, 2).contiguous()[:, :q_len, :, :].to(dtype=out_dtype)
        return self.dropout(out), None


class AutoformerLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        bias = x.mean(dim=1, keepdim=True)
        return x - bias


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, attention: MultiHeadAttentionLayer, *, d_model: int, d_ff: int, moving_avg: int, dropout: float):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_x, attn = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x, _ = self.decomp2(x + y)
        return x, attn


class AutoformerEncoder(nn.Module):
    def __init__(self, layers: list[AutoformerEncoderLayer], *, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = AutoformerLayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        return self.norm(x), attns


class AutoformerDecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionLayer,
        cross_attention: MultiHeadAttentionLayer,
        *,
        d_model: int,
        prob_dim: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=prob_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=None)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, memory, memory, attn_mask=None)[0])
        x, trend2 = self.decomp2(x)
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x, trend3 = self.decomp3(x + y)
        residual_trend = self.projection((trend1 + trend2 + trend3).transpose(1, 2)).transpose(1, 2)
        return x, trend + residual_trend


class AutoformerDecoder(nn.Module):
    def __init__(self, layers: list[AutoformerDecoderLayer], *, d_model: int, prob_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = AutoformerLayerNorm(d_model)
        self.projection = nn.Linear(d_model, prob_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, *, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x, trend = layer(x, memory, trend=trend)
        x = self.norm(x)
        return self.projection(x), trend


class TiDELayerNorm(nn.Module):
    def __init__(self, ndim: int, *, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class TiDEResBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = TiDELayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))
        out = self.dropout(self.fc2(out))
        out = out + self.skip(x)
        return self.norm(out)


class DLinearSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.decomposition = SeriesDecomposition(cfg.moving_avg)
        self.individual = cfg.individual

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(cfg.max_seq_len, 1) for _ in range(cfg.prob_dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(cfg.max_seq_len, 1) for _ in range(cfg.prob_dim)]
            )
            for layer in list(self.linear_seasonal) + list(self.linear_trend):
                nn.init.constant_(layer.weight, 1.0 / cfg.max_seq_len)
                nn.init.zeros_(layer.bias)
        else:
            self.linear_seasonal = nn.Linear(cfg.max_seq_len, 1)
            self.linear_trend = nn.Linear(cfg.max_seq_len, 1)
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / cfg.max_seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / cfg.max_seq_len)
            nn.init.zeros_(self.linear_seasonal.bias)
            nn.init.zeros_(self.linear_trend.bias)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"DLinearSimplexBackbone expects context windows of length {self.cfg.max_seq_len}, "
                f"got {probs.size(1)}"
            )
        x = probs
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)

        if self.individual:
            seasonal_output = seasonal_init.new_zeros(seasonal_init.size(0), self.cfg.prob_dim, 1)
            trend_output = trend_init.new_zeros(trend_init.size(0), self.cfg.prob_dim, 1)
            for idx in range(self.cfg.prob_dim):
                seasonal_output[:, idx, :] = self.linear_seasonal[idx](seasonal_init[:, idx, :])
                trend_output[:, idx, :] = self.linear_trend[idx](trend_init[:, idx, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        logits = (seasonal_output + trend_output).transpose(1, 2)
        return logits


class _TimeMixerScaleHead(nn.Module):
    def __init__(self, context_len: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.context_len = int(context_len)
        self.temporal_proj = nn.Linear(self.context_len, d_model)
        self.mixer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); project each variate's temporal trace with shared weights.
        if x.size(1) != self.context_len:
            raise ValueError(f"expected scale length {self.context_len}, got {x.size(1)}")
        hidden = self.temporal_proj(x.transpose(1, 2))
        hidden = self.mixer(hidden)
        return self.out(hidden).squeeze(-1)


class TimeMixerSimplexBackbone(nn.Module):
    """
    Lightweight TimeMixer-style multiscale decomposition baseline.

    This keeps the existing fixed-window simplex-head contract while adding the
    key TimeMixer ingredients: multiple temporal resolutions, seasonal/trend
    decomposition at each resolution, MLP mixing, and a final softmax head.
    """

    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.scale_lengths = self._build_scale_lengths(
            cfg.max_seq_len,
            cfg.timemixer_levels,
            cfg.timemixer_downsample_factor,
        )
        self.decompositions = nn.ModuleList(SeriesDecomposition(cfg.moving_avg) for _ in self.scale_lengths)
        self.seasonal_heads = nn.ModuleList(
            _TimeMixerScaleHead(length, cfg.d_model, cfg.d_ff, cfg.dropout) for length in self.scale_lengths
        )
        self.trend_heads = nn.ModuleList(
            _TimeMixerScaleHead(length, cfg.d_model, cfg.d_ff, cfg.dropout) for length in self.scale_lengths
        )
        self.seasonal_scale_logits = nn.Parameter(torch.zeros(len(self.scale_lengths)))
        self.trend_scale_logits = nn.Parameter(torch.zeros(len(self.scale_lengths)))
        self.channel_mixer = (
            nn.Sequential(
                nn.LayerNorm(cfg.prob_dim),
                nn.Linear(cfg.prob_dim, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.prob_dim),
            )
            if cfg.timemixer_channel_mixing
            else None
        )

    @staticmethod
    def _build_scale_lengths(max_seq_len: int, levels: int, factor: int) -> list[int]:
        lengths = [int(max_seq_len)]
        current = int(max_seq_len)
        for _ in range(1, int(levels)):
            current = max(1, math.ceil(current / int(factor)))
            lengths.append(current)
        return lengths

    def _multiscale_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        scales = [x]
        current = x
        for target_len in self.scale_lengths[1:]:
            if current.size(1) <= 1:
                downsampled = current
            else:
                kernel = min(self.cfg.timemixer_downsample_factor, current.size(1))
                downsampled = F.avg_pool1d(
                    current.transpose(1, 2),
                    kernel_size=kernel,
                    stride=self.cfg.timemixer_downsample_factor,
                    ceil_mode=True,
                ).transpose(1, 2)
            if downsampled.size(1) != target_len:
                downsampled = F.interpolate(
                    downsampled.transpose(1, 2),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            scales.append(downsampled)
            current = downsampled
        return scales

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"TimeMixer expects context windows of length {self.cfg.max_seq_len}, got {probs.size(1)}"
            )

        x = _temporal_standardize(probs, self.cfg.normalize_input)
        seasonal_outputs = []
        trend_outputs = []
        for idx, scale in enumerate(self._multiscale_inputs(x)):
            seasonal, trend = self.decompositions[idx](scale)
            seasonal_outputs.append(self.seasonal_heads[idx](seasonal))
            trend_outputs.append(self.trend_heads[idx](trend))

        seasonal_stack = torch.stack(seasonal_outputs, dim=1)
        trend_stack = torch.stack(trend_outputs, dim=1)
        seasonal_weights = torch.softmax(self.seasonal_scale_logits, dim=0).view(1, -1, 1)
        trend_weights = torch.softmax(self.trend_scale_logits, dim=0).view(1, -1, 1)
        logits = (seasonal_stack * seasonal_weights).sum(dim=1)
        logits = logits + (trend_stack * trend_weights).sum(dim=1)
        if self.channel_mixer is not None:
            logits = logits + self.channel_mixer(logits)
        return logits.unsqueeze(1)


class ITransformerSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.embedding = InvertedEmbedding(
            prob_dim=cfg.prob_dim,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        self.projection = nn.Linear(cfg.d_model, 1, bias=True)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"ITransformerSimplexBackbone expects context windows of length {self.cfg.max_seq_len}, "
                f"got {probs.size(1)}"
            )
        x = probs
        x = _temporal_standardize(x, self.cfg.normalize_input)

        enc_out = self.embedding(x)
        enc_out = self.encoder(enc_out)
        logits = self.projection(enc_out).transpose(1, 2)
        return logits


class InformerSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.enc_embedding = SequenceEmbedding(
            prob_dim=cfg.prob_dim,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            use_position=True,
        )
        self.dec_embedding = SequenceEmbedding(
            prob_dim=cfg.prob_dim,
            max_seq_len=1,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            use_position=True,
        )
        encoder_layers = [
            InformerEncoderLayer(
                MultiHeadAttentionLayer(
                    ProbSparseAttention(
                        mask_flag=False,
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layer)
        ]
        conv_layers = (
            [InformerConvLayer(cfg.d_model) for _ in range(cfg.n_layer - 1)]
            if cfg.informer_distil and cfg.n_layer > 1
            else None
        )
        self.encoder = InformerEncoder(encoder_layers, conv_layers=conv_layers)
        decoder_layers = [
            InformerDecoderLayer(
                MultiHeadAttentionLayer(
                    ProbSparseAttention(
                        mask_flag=True,
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                MultiHeadAttentionLayer(
                    ProbSparseAttention(
                        mask_flag=False,
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.decoder_layers)
        ]
        self.decoder = InformerDecoder(decoder_layers, d_model=cfg.d_model, prob_dim=cfg.prob_dim)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"Informer expects context windows of length {self.cfg.max_seq_len}, got {probs.size(1)}"
            )
        x = _temporal_standardize(probs, self.cfg.normalize_input)
        enc_out = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        dec_in = x[:, -1:, :]
        dec_out = self.dec_embedding(dec_in)
        return self.decoder(dec_out, enc_out)


class AutoformerSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.decomposition = SeriesDecomposition(cfg.moving_avg)
        self.enc_embedding = SequenceEmbedding(
            prob_dim=cfg.prob_dim,
            max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            use_position=False,
        )
        self.dec_embedding = SequenceEmbedding(
            prob_dim=cfg.prob_dim,
            max_seq_len=1,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            use_position=False,
        )
        encoder_layers = [
            AutoformerEncoderLayer(
                MultiHeadAttentionLayer(
                    AutoCorrelationAttention(
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                moving_avg=cfg.moving_avg,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layer)
        ]
        self.encoder = AutoformerEncoder(encoder_layers, d_model=cfg.d_model)
        decoder_layers = [
            AutoformerDecoderLayer(
                MultiHeadAttentionLayer(
                    AutoCorrelationAttention(
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                MultiHeadAttentionLayer(
                    AutoCorrelationAttention(
                        factor=cfg.attention_factor,
                        attention_dropout=cfg.dropout,
                    ),
                    d_model=cfg.d_model,
                    n_heads=cfg.n_head,
                ),
                d_model=cfg.d_model,
                prob_dim=cfg.prob_dim,
                d_ff=cfg.d_ff,
                moving_avg=cfg.moving_avg,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.decoder_layers)
        ]
        self.decoder = AutoformerDecoder(decoder_layers, d_model=cfg.d_model, prob_dim=cfg.prob_dim)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"Autoformer expects context windows of length {self.cfg.max_seq_len}, got {probs.size(1)}"
            )
        x = _temporal_standardize(probs, self.cfg.normalize_input)
        seasonal_init, _ = self.decomposition(x)
        enc_out = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        seasonal_query = seasonal_init[:, -1:, :]
        trend_query = x.mean(dim=1, keepdim=True)
        dec_out = self.dec_embedding(seasonal_query)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_query)
        return seasonal_part + trend_part


class TiDESimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        self.time_features = nn.Parameter(torch.zeros(1, cfg.max_seq_len + 1, cfg.tide_feature_dim))
        nn.init.trunc_normal_(self.time_features, std=0.02)

        flat_input_dim = cfg.max_seq_len * cfg.prob_dim + (cfg.max_seq_len + 1) * cfg.tide_feature_dim
        encoder_blocks = [TiDEResBlock(flat_input_dim, cfg.d_ff, cfg.d_model, dropout=cfg.dropout)]
        encoder_blocks.extend(
            TiDEResBlock(cfg.d_model, cfg.d_ff, cfg.d_model, dropout=cfg.dropout) for _ in range(cfg.n_layer - 1)
        )
        decoder_blocks = [
            TiDEResBlock(cfg.d_model, cfg.d_ff, cfg.d_model, dropout=cfg.dropout)
            for _ in range(max(cfg.decoder_layers - 1, 0))
        ]
        decoder_blocks.append(TiDEResBlock(cfg.d_model, cfg.d_ff, cfg.prob_dim, dropout=cfg.dropout))
        self.encoder = nn.Sequential(*encoder_blocks)
        self.decoder = nn.Sequential(*decoder_blocks)
        self.temporal_decoder = TiDEResBlock(
            cfg.prob_dim + cfg.tide_feature_dim,
            cfg.d_ff,
            cfg.prob_dim,
            dropout=cfg.dropout,
        )
        self.residual_proj = nn.Linear(cfg.max_seq_len, 1)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"TiDE expects context windows of length {self.cfg.max_seq_len}, got {probs.size(1)}"
            )
        x = _temporal_standardize(probs, self.cfg.normalize_input)
        batch_size = x.size(0)
        time_features = self.time_features.expand(batch_size, -1, -1)
        enc_input = torch.cat([x.reshape(batch_size, -1), time_features.reshape(batch_size, -1)], dim=-1)
        hidden = self.encoder(enc_input)
        decoded = self.decoder(hidden).unsqueeze(1)
        future_feature = time_features[:, -1:, :]
        logits = self.temporal_decoder(torch.cat([decoded, future_feature], dim=-1))
        residual = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        return logits + residual


class RecurrentSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig", *, rnn_type: str):
        super().__init__()
        self.cfg = cfg
        normalized = rnn_type.strip().lower()
        if normalized == "gru":
            rnn_cls = nn.GRU
        elif normalized == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError(f"unsupported recurrent backbone={rnn_type!r}")

        self.rnn = rnn_cls(
            input_size=cfg.prob_dim,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layer,
            dropout=cfg.dropout if cfg.n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(cfg.d_model, cfg.prob_dim)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"{self.cfg.backbone} expects context windows of length {self.cfg.max_seq_len}, "
                f"got {probs.size(1)}"
            )
        x = _temporal_standardize(probs, self.cfg.normalize_input)
        enc_out, _ = self.rnn(x)
        logits = self.projection(enc_out[:, -1, :]).unsqueeze(1)
        return logits


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = (kernel_size - 1) * dilation
        self.chomp = padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        return out


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TCNSimplexBackbone(nn.Module):
    def __init__(self, cfg: "TSLibSimplexConfig"):
        super().__init__()
        self.cfg = cfg
        blocks = []
        in_channels = cfg.prob_dim
        for layer_idx in range(cfg.n_layer):
            dilation = int(cfg.tcn_dilation_base ** layer_idx)
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=cfg.d_model,
                    kernel_size=cfg.tcn_kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                )
            )
            in_channels = cfg.d_model
        self.network = nn.Sequential(*blocks)
        self.projection = nn.Linear(cfg.d_model, cfg.prob_dim)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        if probs.size(1) != self.cfg.max_seq_len:
            raise ValueError(
                f"TCN expects context windows of length {self.cfg.max_seq_len}, got {probs.size(1)}"
            )
        x = _temporal_standardize(probs, self.cfg.normalize_input).transpose(1, 2)
        enc_out = self.network(x)
        logits = self.projection(enc_out[:, :, -1]).unsqueeze(1)
        return logits


@dataclass
class TSLibSimplexConfig:
    backbone: str
    prob_dim: int
    max_seq_len: int = 512
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    simplex_eps: float = 1e-8
    loss_type: str = "ce"
    normalize_input: bool = True
    moving_avg: int = 25
    individual: bool = False
    window_batch_size: int = 256
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    decoder_layers: int = 1
    attention_factor: int = 5
    informer_distil: bool = True
    tide_feature_dim: int = 2
    timemixer_levels: int = 3
    timemixer_downsample_factor: int = 2
    timemixer_channel_mixing: bool = True

    def __post_init__(self) -> None:
        self.backbone = self.backbone.strip().lower()
        if self.backbone not in {
            "dlinear",
            "timemixer",
            "itransformer",
            "informer",
            "autoformer",
            "tide",
            "tcn",
            "gru",
            "lstm",
        }:
            raise ValueError(
                "backbone must be one of {'dlinear', 'timemixer', 'itransformer', "
                "'informer', 'autoformer', 'tide', 'tcn', 'gru', 'lstm'}"
            )
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.backbone in {"itransformer", "informer", "autoformer"} and self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.simplex_eps <= 0.0:
            raise ValueError("simplex_eps must be positive")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.moving_avg <= 0:
            raise ValueError("moving_avg must be positive")
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        if self.tcn_kernel_size <= 0:
            raise ValueError("tcn_kernel_size must be positive")
        if self.tcn_dilation_base <= 0:
            raise ValueError("tcn_dilation_base must be positive")
        if self.decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        if self.attention_factor <= 0:
            raise ValueError("attention_factor must be positive")
        if self.tide_feature_dim <= 0:
            raise ValueError("tide_feature_dim must be positive")
        if self.timemixer_levels <= 0:
            raise ValueError("timemixer_levels must be positive")
        if self.timemixer_downsample_factor <= 1:
            raise ValueError("timemixer_downsample_factor must be > 1")
        self.normalize_input = _bool_env_like(self.normalize_input)
        self.individual = _bool_env_like(self.individual)
        self.informer_distil = _bool_env_like(self.informer_distil)
        self.timemixer_channel_mixing = _bool_env_like(self.timemixer_channel_mixing)


class TSLibSimplexBaseline(nn.Module):
    """
    Minimal deep forecasting baselines with shared simplex output handling.

    We use each backbone in its native fixed-window forecasting mode: for every
    time step t in the batch, build a causal context window ending at t, predict
    the next step, and stitch those one-step forecasts back together. That keeps
    the evaluation honest for next-step prediction and avoids leaking future
    observations from later positions in the same block.
    """

    def __init__(self, cfg: TSLibSimplexConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "dlinear":
            self.backbone = DLinearSimplexBackbone(cfg)
        elif cfg.backbone == "timemixer":
            self.backbone = TimeMixerSimplexBackbone(cfg)
        elif cfg.backbone == "itransformer":
            self.backbone = ITransformerSimplexBackbone(cfg)
        elif cfg.backbone == "informer":
            self.backbone = InformerSimplexBackbone(cfg)
        elif cfg.backbone == "autoformer":
            self.backbone = AutoformerSimplexBackbone(cfg)
        elif cfg.backbone == "tide":
            self.backbone = TiDESimplexBackbone(cfg)
        elif cfg.backbone == "tcn":
            self.backbone = TCNSimplexBackbone(cfg)
        elif cfg.backbone == "gru":
            self.backbone = RecurrentSimplexBackbone(cfg, rnn_type="gru")
        elif cfg.backbone == "lstm":
            self.backbone = RecurrentSimplexBackbone(cfg, rnn_type="lstm")
        else:
            raise ValueError(f"unsupported backbone={cfg.backbone!r}")

    def _forecast_logits(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        batch_size, total_steps, prob_dim = probs.shape
        logits_chunks = []
        for start, end, chunk_windows in _iter_context_window_chunks(
            probs,
            context_len=self.cfg.max_seq_len,
            window_batch_size=self.cfg.window_batch_size,
        ):
            chunk_logits = self.backbone(chunk_windows)
            if chunk_logits.ndim != 3 or chunk_logits.size(1) != 1 or chunk_logits.size(2) != prob_dim:
                raise RuntimeError(
                    f"{self.cfg.backbone} backbone must return shape (N,1,D); got {tuple(chunk_logits.shape)}"
                )
            logits_chunks.append(chunk_logits[:, 0, :].view(batch_size, end - start, prob_dim))

        return torch.cat(logits_chunks, dim=1) if logits_chunks else probs.new_zeros(batch_size, total_steps, prob_dim)

    def forward(
        self,
        probs: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        ablation: Optional[Mapping[str, object]] = None,
        return_dict: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        del return_aux
        if ablation:
            raise ValueError(f"runtime ablations are not supported for MODEL={self.cfg.backbone}")

        logits = self._forecast_logits(probs)
        pred = torch.softmax(logits, dim=-1)
        if not return_dict:
            return pred

        out: Dict[str, torch.Tensor] = {
            "pred": pred,
            "logits": logits,
        }

        if targets is not None:
            main_loss = distribution_loss(
                pred,
                targets,
                loss_type=self.cfg.loss_type,
                eps=self.cfg.simplex_eps,
                reduction="mean",
                mask=loss_mask,
            )
            zero = main_loss.new_zeros(())
            out.update(
                {
                    "loss": main_loss,
                    "main_loss": main_loss,
                    "aux_prior_loss": zero,
                    "transport_reg_loss": zero,
                }
            )

        return out
