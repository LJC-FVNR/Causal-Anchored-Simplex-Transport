from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Callable, Dict, Iterator, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ilr_var import (
    _align_fit_sequence,
    _helmert_basis,
    _ilr_to_simplex,
    _iter_sequences_from_split_payload,
    _normalize_simplex,
    _simplex_to_ilr,
)
from layer_perm_mixed_ablation import distribution_loss
from tasks.base import ModelFitSequence


@dataclass
class AnalogSuccessorConfig:
    prob_dim: int
    window: int = 8
    stride: int = 1
    k: int = 16
    coordinate_mode: str = "ilr"
    include_deltas: bool = True
    value_mode: str = "delta"
    blend_weight: float = 0.5
    distance: str = "euclidean"
    temperature: float = 0.1
    max_memory_items: int = 200000
    memory_split: str = "train"
    memory_subsample_seed: int = 42
    query_chunk_size: int = 256
    memory_chunk_size: int = 16384
    simplex_eps: float = 1e-8
    loss_type: str = "ce"
    fallback: str = "persistence"
    fitted_num_samples: int = 0
    effective_memory_items: int = 0
    key_dim: int = 0

    def __post_init__(self) -> None:
        self.coordinate_mode = self.coordinate_mode.strip().lower()
        self.value_mode = self.value_mode.strip().lower()
        self.distance = self.distance.strip().lower()
        self.memory_split = self.memory_split.strip().lower() or "train"
        self.fallback = self.fallback.strip().lower()

        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.coordinate_mode not in {"raw", "clr", "ilr", "raw_ilr"}:
            raise ValueError("coordinate_mode must be one of {'raw', 'clr', 'ilr', 'raw_ilr'}")
        if self.value_mode not in {"delta", "successor", "blend"}:
            raise ValueError("value_mode must be one of {'delta', 'successor', 'blend'}")
        if not (0.0 <= self.blend_weight <= 1.0):
            raise ValueError("blend_weight must be in [0, 1]")
        if self.distance not in {"euclidean", "cosine"}:
            raise ValueError("distance must be one of {'euclidean', 'cosine'}")
        if self.memory_split != "train":
            raise ValueError("ANALOG_MEMORY_SPLIT must be 'train' to avoid validation/test leakage")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.max_memory_items < 0:
            raise ValueError("max_memory_items must be >= 0; use 0 for unlimited")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.memory_chunk_size <= 0:
            raise ValueError("memory_chunk_size must be positive")
        if self.simplex_eps <= 0.0:
            raise ValueError("simplex_eps must be positive")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.fallback != "persistence":
            raise ValueError("currently only fallback='persistence' is supported")


class AnalogSuccessorBaseline(nn.Module):
    """
    Non-neural analog forecasting baseline for simplex-valued sequences.

    A train-split memory bank stores context-window keys and observed successors.
    At inference time, each context window queries the memory bank, then predicts
    either a weighted successor distribution, a weighted ilr delta from the
    current distribution, or a convex blend of both.
    """

    def __init__(
        self,
        cfg: AnalogSuccessorConfig,
        *,
        basis: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_anchors: torch.Tensor,
        memory_values: torch.Tensor,
        memory_deltas: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("basis", basis.to(torch.float32))
        self.register_buffer("memory_keys", memory_keys.to(torch.float32))
        self.register_buffer("memory_anchors", memory_anchors.to(torch.float32))
        self.register_buffer("memory_values", memory_values.to(torch.float32))
        self.register_buffer("memory_deltas", memory_deltas.to(torch.float32))

    @classmethod
    def fit_from_split_payload(cls, cfg: AnalogSuccessorConfig, split_payload) -> "AnalogSuccessorBaseline":
        sequence_factory = lambda: _iter_sequences_from_split_payload(split_payload)
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def fit_from_task(cls, cfg: AnalogSuccessorConfig, task) -> "AnalogSuccessorBaseline":
        sequence_factory = lambda: task.iter_model_fit_sequences(cfg.memory_split, model_name="analog_successor")
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def _fit_from_sequence_factory(
        cls,
        cfg: AnalogSuccessorConfig,
        sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    ) -> "AnalogSuccessorBaseline":
        basis = _helmert_basis(cfg.prob_dim, dtype=torch.float64)
        key_dim = cls._key_dim(cfg)
        keys: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        rng = random.Random(int(cfg.memory_subsample_seed))
        seen = 0

        def add_item(key: torch.Tensor, anchor: torch.Tensor, value: torch.Tensor, delta: torch.Tensor) -> None:
            nonlocal seen
            seen += 1
            if cfg.max_memory_items == 0 or len(keys) < cfg.max_memory_items:
                keys.append(key.detach().to(torch.float32).cpu())
                anchors.append(anchor.detach().to(torch.float32).cpu())
                values.append(value.detach().to(torch.float32).cpu())
                deltas.append(delta.detach().to(torch.float32).cpu())
                return

            replace_idx = rng.randrange(seen)
            if replace_idx < cfg.max_memory_items:
                keys[replace_idx] = key.detach().to(torch.float32).cpu()
                anchors[replace_idx] = anchor.detach().to(torch.float32).cpu()
                values[replace_idx] = value.detach().to(torch.float32).cpu()
                deltas[replace_idx] = delta.detach().to(torch.float32).cpu()

        for example in sequence_factory():
            context, targets, mask = _align_fit_sequence(example)
            if context is None or targets is None or mask is None:
                continue
            if context.size(-1) != cfg.prob_dim:
                raise ValueError(
                    f"fit sequence prob_dim mismatch: expected {cfg.prob_dim}, got {context.size(-1)}"
                )
            if context.size(0) == 0:
                continue

            context = _normalize_simplex(context.to(dtype=basis.dtype), eps=cfg.simplex_eps)
            targets = _normalize_simplex(targets.to(dtype=basis.dtype), eps=cfg.simplex_eps)
            coords = cls._coordinates(context, basis, cfg)
            z_context = _simplex_to_ilr(context, basis, cfg.simplex_eps)
            z_targets = _simplex_to_ilr(targets, basis, cfg.simplex_eps)

            valid_idx = torch.nonzero(mask.to(torch.bool), as_tuple=False).squeeze(-1)
            if cfg.stride > 1:
                valid_idx = valid_idx[valid_idx.remainder(int(cfg.stride)) == 0]
            for idx in valid_idx.tolist():
                key = cls._key_at(coords, int(idx), cfg)
                anchor = context[int(idx)]
                value = targets[int(idx)]
                delta = z_targets[int(idx)] - z_context[int(idx)]
                add_item(key, anchor, value, delta)

        if keys:
            memory_keys = torch.stack(keys, dim=0).contiguous()
            memory_anchors = torch.stack(anchors, dim=0).contiguous()
            memory_values = torch.stack(values, dim=0).contiguous()
            memory_deltas = torch.stack(deltas, dim=0).contiguous()
        else:
            memory_keys = torch.empty((0, key_dim), dtype=torch.float32)
            memory_anchors = torch.empty((0, cfg.prob_dim), dtype=torch.float32)
            memory_values = torch.empty((0, cfg.prob_dim), dtype=torch.float32)
            memory_deltas = torch.empty((0, cfg.prob_dim - 1), dtype=torch.float32)

        fitted_cfg = replace(
            cfg,
            fitted_num_samples=seen,
            effective_memory_items=int(memory_keys.size(0)),
            key_dim=key_dim,
        )
        return cls(
            fitted_cfg,
            basis=basis,
            memory_keys=memory_keys,
            memory_anchors=memory_anchors,
            memory_values=memory_values,
            memory_deltas=memory_deltas,
        )

    @staticmethod
    def _coordinate_dim(cfg: AnalogSuccessorConfig) -> int:
        if cfg.coordinate_mode == "raw":
            return cfg.prob_dim
        if cfg.coordinate_mode in {"clr", "ilr"}:
            return cfg.prob_dim if cfg.coordinate_mode == "clr" else cfg.prob_dim - 1
        if cfg.coordinate_mode == "raw_ilr":
            return cfg.prob_dim + cfg.prob_dim - 1
        raise RuntimeError(f"unsupported coordinate_mode={cfg.coordinate_mode!r}")

    @classmethod
    def _key_dim(cls, cfg: AnalogSuccessorConfig) -> int:
        coord_dim = cls._coordinate_dim(cfg)
        delta_steps = max(int(cfg.window) - 1, 0) if cfg.include_deltas else 0
        return int(cfg.window) * coord_dim + delta_steps * coord_dim

    @staticmethod
    def _coordinates(probs: torch.Tensor, basis: torch.Tensor, cfg: AnalogSuccessorConfig) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=cfg.simplex_eps).to(dtype=basis.dtype)
        if cfg.coordinate_mode == "raw":
            return probs

        log_probs = probs.clamp_min(cfg.simplex_eps).log()
        clr = log_probs - log_probs.mean(dim=-1, keepdim=True)
        if cfg.coordinate_mode == "clr":
            return clr

        ilr = _simplex_to_ilr(probs, basis.to(device=probs.device, dtype=probs.dtype), cfg.simplex_eps)
        if cfg.coordinate_mode == "ilr":
            return ilr
        if cfg.coordinate_mode == "raw_ilr":
            return torch.cat([probs, ilr], dim=-1)
        raise RuntimeError(f"unsupported coordinate_mode={cfg.coordinate_mode!r}")

    @staticmethod
    def _key_at(coords: torch.Tensor, index: int, cfg: AnalogSuccessorConfig) -> torch.Tensor:
        if coords.ndim != 2:
            raise ValueError(f"expected coords with shape (T,F), got {tuple(coords.shape)}")
        if coords.size(0) <= 0:
            raise ValueError("cannot build analog key from an empty sequence")

        window = int(cfg.window)
        start = index - window + 1
        pieces = []
        for pos in range(start, index + 1):
            pieces.append(coords[max(pos, 0)])
        window_values = torch.stack(pieces, dim=0)
        features = [window_values.reshape(-1)]
        if cfg.include_deltas and window > 1:
            features.append((window_values[1:] - window_values[:-1]).reshape(-1))
        return torch.cat(features, dim=0).contiguous()

    @classmethod
    def _query_keys(cls, probs: torch.Tensor, basis: torch.Tensor, cfg: AnalogSuccessorConfig) -> torch.Tensor:
        batch_size, steps, _ = probs.shape
        coords = cls._coordinates(probs.reshape(-1, cfg.prob_dim), basis, cfg).view(batch_size, steps, -1)
        keys = []
        for batch_idx in range(batch_size):
            for step in range(steps):
                keys.append(cls._key_at(coords[batch_idx], step, cfg))
        if not keys:
            return probs.new_empty((0, cls._key_dim(cfg)))
        return torch.stack(keys, dim=0).to(device=probs.device, dtype=probs.dtype).contiguous()

    @staticmethod
    def _distance_matrix(query: torch.Tensor, memory: torch.Tensor, *, distance: str) -> torch.Tensor:
        if distance == "euclidean":
            q_norm = query.square().sum(dim=1, keepdim=True)
            m_norm = memory.square().sum(dim=1).view(1, -1)
            dist = q_norm + m_norm - 2.0 * (query @ memory.transpose(0, 1))
            return dist.clamp_min(0.0)
        if distance == "cosine":
            q = F.normalize(query, dim=1, eps=1e-12)
            m = F.normalize(memory, dim=1, eps=1e-12)
            return (1.0 - q @ m.transpose(0, 1)).clamp_min(0.0)
        raise RuntimeError(f"unsupported distance={distance!r}")

    def _lookup(self, query_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory_size = int(self.memory_keys.size(0))
        if memory_size <= 0:
            raise RuntimeError("cannot query an empty analog memory")

        k_eff = min(int(self.cfg.k), memory_size)
        all_values = []
        all_deltas = []
        memory_keys = self.memory_keys.to(device=query_keys.device, dtype=query_keys.dtype)
        memory_values = self.memory_values.to(device=query_keys.device, dtype=query_keys.dtype)
        memory_deltas = self.memory_deltas.to(device=query_keys.device, dtype=query_keys.dtype)

        for query_start in range(0, query_keys.size(0), int(self.cfg.query_chunk_size)):
            query = query_keys[query_start : query_start + int(self.cfg.query_chunk_size)]
            best_dist = query.new_full((query.size(0), k_eff), float("inf"))
            best_idx = torch.zeros((query.size(0), k_eff), device=query.device, dtype=torch.long)

            for mem_start in range(0, memory_size, int(self.cfg.memory_chunk_size)):
                memory = memory_keys[mem_start : mem_start + int(self.cfg.memory_chunk_size)]
                dist = self._distance_matrix(query, memory, distance=self.cfg.distance)
                chunk_k = min(k_eff, dist.size(1))
                dist_top, idx_top = torch.topk(dist, chunk_k, dim=1, largest=False)
                idx_top = idx_top + mem_start

                merged_dist = torch.cat([best_dist, dist_top], dim=1)
                merged_idx = torch.cat([best_idx, idx_top], dim=1)
                best_dist, order = torch.topk(merged_dist, k_eff, dim=1, largest=False)
                best_idx = torch.gather(merged_idx, 1, order)

            weights = torch.softmax(-best_dist / float(self.cfg.temperature), dim=1)
            selected_values = memory_values.index_select(0, best_idx.reshape(-1)).view(query.size(0), k_eff, -1)
            selected_deltas = memory_deltas.index_select(0, best_idx.reshape(-1)).view(query.size(0), k_eff, -1)
            all_values.append((weights.unsqueeze(-1) * selected_values).sum(dim=1))
            all_deltas.append((weights.unsqueeze(-1) * selected_deltas).sum(dim=1))

        return torch.cat(all_values, dim=0), torch.cat(all_deltas, dim=0)

    def _predict_sequence(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        if int(self.memory_keys.size(0)) <= 0:
            return probs.clone()

        batch_size, steps, prob_dim = probs.shape
        basis = self.basis.to(device=probs.device, dtype=probs.dtype)
        query_keys = self._query_keys(probs, basis, self.cfg)
        successor_pred, delta = self._lookup(query_keys)

        if self.cfg.value_mode in {"delta", "blend"}:
            last_z = _simplex_to_ilr(
                probs.reshape(-1, prob_dim),
                basis,
                self.cfg.simplex_eps,
            )
            delta_pred = _ilr_to_simplex(last_z + delta, basis, self.cfg.simplex_eps)
        else:
            delta_pred = successor_pred

        if self.cfg.value_mode == "successor":
            pred = successor_pred
        elif self.cfg.value_mode == "delta":
            pred = delta_pred
        elif self.cfg.value_mode == "blend":
            blend = float(self.cfg.blend_weight)
            pred = blend * successor_pred + (1.0 - blend) * delta_pred
        else:
            raise RuntimeError(f"unsupported value_mode={self.cfg.value_mode!r}")

        fallback = probs.reshape(-1, prob_dim)
        bad = ~torch.isfinite(pred).all(dim=-1, keepdim=True)
        pred = torch.where(bad, fallback, pred)
        pred = _normalize_simplex(pred, eps=self.cfg.simplex_eps)
        return pred.view(batch_size, steps, prob_dim)

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
            raise ValueError("runtime ablations are not supported for MODEL=analog_successor")

        pred = self._predict_sequence(probs)
        if not return_dict:
            return pred

        logits = pred.clamp_min(self.cfg.simplex_eps).log()
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
