from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .ilr_var import _align_fit_sequence, _iter_sequences_from_split_payload
from layer_perm_mixed_ablation import distribution_loss
from tasks.base import ModelFitSequence


def _normalize_simplex(probs: torch.Tensor, eps: float) -> torch.Tensor:
    if not torch.is_floating_point(probs):
        probs = probs.float()
    probs = probs.clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def _strictly_positive_simplex(probs: torch.Tensor, eps: float) -> torch.Tensor:
    probs = _normalize_simplex(probs, eps=eps)
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _choose_reference_index(sequence: torch.Tensor) -> int:
    return int(torch.argmax(sequence.mean(dim=0)).item())


def _simplex_to_alr(probs: torch.Tensor, ref_index: int, eps: float) -> torch.Tensor:
    probs = _strictly_positive_simplex(probs, eps)
    ref = probs[..., ref_index : ref_index + 1]
    left = probs[..., :ref_index]
    right = probs[..., ref_index + 1 :]
    others = torch.cat([left, right], dim=-1)
    return others.log() - ref.log()


def _alr_to_simplex(coords: torch.Tensor, ref_index: int, prob_dim: int, eps: float) -> torch.Tensor:
    zeros = torch.zeros(*coords.shape[:-1], 1, dtype=coords.dtype, device=coords.device)
    if ref_index == 0:
        logits = torch.cat([zeros, coords], dim=-1)
    elif ref_index == prob_dim - 1:
        logits = torch.cat([coords, zeros], dim=-1)
    else:
        logits = torch.cat([coords[..., :ref_index], zeros, coords[..., ref_index:]], dim=-1)
    return _normalize_simplex(torch.softmax(logits, dim=-1), eps=eps)


def _single_sequence_from_factory(
    sequence_factory,
    *,
    prob_dim: int,
    simplex_eps: float,
) -> Tuple[torch.Tensor, int]:
    full_sequences = []

    for example in sequence_factory():
        context, targets, mask = _align_fit_sequence(example)
        if context is None or targets is None:
            continue
        if context.size(-1) != prob_dim or targets.size(-1) != prob_dim:
            raise ValueError(
                f"B-DARMA prob_dim mismatch: expected {prob_dim}, got {context.size(-1)} / {targets.size(-1)}"
            )
        if mask is not None and (mask.ndim != 1 or not torch.all(mask)):
            raise ValueError(
                "Canonical MODEL=bdarma currently requires a single contiguous unmasked sequence. "
                "Use MODEL=bdarma_map for pooled or masked benchmark data."
            )

        full = torch.cat([context[:1], targets], dim=0).to(torch.float64)
        full_sequences.append(_strictly_positive_simplex(full, simplex_eps))

    if len(full_sequences) != 1:
        raise ValueError(
            f"Canonical MODEL=bdarma currently supports exactly one training sequence, found {len(full_sequences)}. "
            "Use MODEL=bdarma_map for pooled multi-sequence data."
        )

    sequence = full_sequences[0].contiguous()
    if sequence.size(0) < 3:
        raise ValueError("MODEL=bdarma requires at least 3 observations in the fit sequence")
    return sequence, _choose_reference_index(sequence)


def _stan_support_paths() -> Tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    stan_file = here / "stan" / "bdarma_informative_single_series.stan"
    r_file = here / "stan" / "fit_bdarma_informative.R"
    if not stan_file.exists() or not r_file.exists():
        raise FileNotFoundError("Missing vendored B-DARMA Stan support files")
    return stan_file, r_file


@dataclass
class BDARMAConfig:
    prob_dim: int
    ar_order: int = 1
    ma_order: int = 0
    loss_type: str = "ce"
    fit_split: str = "train"
    simplex_eps: float = 1e-6
    prior: str = "informative"
    chains: int = 4
    parallel_chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.9
    max_treedepth: int = 11
    init: float = 0.3
    seed: int = 123
    cache_dir: str = "outputs/bdarma_cache"
    fit_timeout_sec: int = 86400
    fitted_num_samples: int = 0
    effective_ar_order: int = 1
    effective_ma_order: int = 0
    reference_index: int = 0
    phi_intercept: float = 7.0

    def __post_init__(self) -> None:
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.ar_order < 0 or self.ma_order < 0:
            raise ValueError("ar_order and ma_order must be >= 0")
        if self.ar_order + self.ma_order < 1:
            raise ValueError("MODEL=bdarma requires ar_order + ma_order >= 1")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.prior != "informative":
            raise ValueError("Canonical MODEL=bdarma currently supports prior='informative' only")
        if self.chains < 1 or self.parallel_chains < 1:
            raise ValueError("chains and parallel_chains must be >= 1")
        if self.iter_warmup < 1 or self.iter_sampling < 1:
            raise ValueError("iter_warmup and iter_sampling must be >= 1")
        if self.fit_timeout_sec < 1:
            raise ValueError("fit_timeout_sec must be >= 1")
        if not (0 <= self.reference_index < self.prob_dim):
            raise ValueError(f"reference_index must be in [0, {self.prob_dim}), got {self.reference_index}")


class BDARMABaseline(nn.Module):
    """
    Canonical single-series B-DARMA fit via Stan.

    This backend uses a faithful covariate-free special case of the official
    B-DARMA Stan recursion with informative priors. To avoid over-claiming on
    panel benchmarks, it intentionally supports exactly one contiguous training
    sequence. Multi-sequence pooled data should use MODEL=bdarma_map instead.
    """

    def __init__(
        self,
        cfg: BDARMAConfig,
        *,
        ar_coefficients: torch.Tensor,
        ma_coefficients: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.reference_index = int(cfg.reference_index)
        self.register_buffer("ar_coefficients", ar_coefficients.to(torch.float32))
        self.register_buffer("ma_coefficients", ma_coefficients.to(torch.float32))
        self.phi_intercept = float(cfg.phi_intercept)

    @classmethod
    def fit_from_split_payload(cls, cfg: BDARMAConfig, split_payload) -> "BDARMABaseline":
        sequence_factory = lambda: _iter_sequences_from_split_payload(split_payload)
        return cls._fit_from_sequence_factory(cfg, sequence_factory, task_name="split_payload")

    @classmethod
    def fit_from_task(cls, cfg: BDARMAConfig, task) -> "BDARMABaseline":
        sequence_factory = lambda: task.iter_model_fit_sequences(cfg.fit_split, model_name="bdarma")
        return cls._fit_from_sequence_factory(cfg, sequence_factory, task_name=task.name)

    @classmethod
    def _fit_from_sequence_factory(cls, cfg: BDARMAConfig, sequence_factory, *, task_name: str) -> "BDARMABaseline":
        sequence, ref_index = _single_sequence_from_factory(
            sequence_factory,
            prob_dim=cfg.prob_dim,
            simplex_eps=cfg.simplex_eps,
        )
        cache_path = cls._cache_path(cfg, task_name=task_name, sequence=sequence, ref_index=ref_index)
        cached = cls._load_or_fit_cached(cfg, cache_path=cache_path, sequence=sequence, ref_index=ref_index)
        fitted_cfg = replace(
            cfg,
            fitted_num_samples=int(cached["num_samples"]),
            effective_ar_order=int(cached["P"]),
            effective_ma_order=int(cached["Q"]),
            reference_index=int(cached["ref"]) - 1,
            phi_intercept=float(cached["phi_intercept"]),
        )
        return cls(
            fitted_cfg,
            ar_coefficients=torch.tensor(cached["A"], dtype=torch.float32),
            ma_coefficients=torch.tensor(cached["B"], dtype=torch.float32),
        )

    @staticmethod
    def _cache_path(cfg: BDARMAConfig, *, task_name: str, sequence: torch.Tensor, ref_index: int) -> Path:
        cache_root = Path(cfg.cache_dir).expanduser().resolve()
        digest = hashlib.sha1()
        digest.update(task_name.encode("utf-8"))
        digest.update(str(cfg.ar_order).encode("utf-8"))
        digest.update(str(cfg.ma_order).encode("utf-8"))
        digest.update(str(cfg.prior).encode("utf-8"))
        digest.update(str(ref_index).encode("utf-8"))
        digest.update(sequence.detach().cpu().contiguous().numpy().tobytes())
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"bdarma_{digest.hexdigest()[:20]}.json"

    @classmethod
    def _load_or_fit_cached(
        cls,
        cfg: BDARMAConfig,
        *,
        cache_path: Path,
        sequence: torch.Tensor,
        ref_index: int,
    ) -> Dict[str, object]:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        lock_path = cache_path.with_suffix(".lock")
        deadline = time.time() + float(cfg.fit_timeout_sec)
        acquired = False

        while time.time() < deadline:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"pid={os.getpid()}\n")
                acquired = True
                break
            except FileExistsError:
                if cache_path.exists():
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                time.sleep(5.0)

        if not acquired:
            raise TimeoutError(f"Timed out waiting for cached B-DARMA fit at {cache_path}")

        try:
            result = cls._run_stan_fit(cfg, sequence=sequence, ref_index=ref_index, output_path=cache_path)
            cache_path.write_text(json.dumps(result), encoding="utf-8")
            return result
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _run_stan_fit(
        cfg: BDARMAConfig,
        *,
        sequence: torch.Tensor,
        ref_index: int,
        output_path: Path,
    ) -> Dict[str, object]:
        if "SLURM_JOB_ID" not in os.environ and os.environ.get("BDARMA_ALLOW_HEADNODE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(
                "Canonical MODEL=bdarma runs through the compute-node R/cmdstanr module. "
                "Launch it from sbatch/salloc, or set BDARMA_ALLOW_HEADNODE=1 if you have your own R setup."
            )

        stan_file, r_file = _stan_support_paths()
        work_dir = output_path.parent / f".bdarma_fit_{output_path.stem}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "input.json"
        output_json = work_dir / "output.json"

        payload = {
            "T": int(sequence.size(0)),
            "C": int(sequence.size(1)),
            "Y": sequence.detach().cpu().tolist(),
            "ref": int(ref_index) + 1,
            "P": int(cfg.ar_order),
            "Q": int(cfg.ma_order),
            "seed": int(cfg.seed),
            "chains": int(cfg.chains),
            "parallel_chains": int(cfg.parallel_chains),
            "iter_warmup": int(cfg.iter_warmup),
            "iter_sampling": int(cfg.iter_sampling),
            "adapt_delta": float(cfg.adapt_delta),
            "max_treedepth": int(cfg.max_treedepth),
            "init": float(cfg.init),
        }
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        shell_cmd = (
            "module load r/4.4.1-cuda >/dev/null 2>&1 && "
            f"Rscript {r_file} {input_path} {output_json} {stan_file}"
        )
        proc = subprocess.run(
            ["/bin/bash", "-lc", shell_cmd],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(cfg.fit_timeout_sec),
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Canonical B-DARMA Stan fit failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if not output_json.exists():
            raise RuntimeError("Canonical B-DARMA Stan fit did not produce an output JSON file")

        result = json.loads(output_json.read_text(encoding="utf-8"))
        if int(result["ref"]) != int(ref_index) + 1:
            raise RuntimeError("B-DARMA Stan output returned an inconsistent reference index")
        return result

    def _compute_observed_state(self, alr_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length, alr_dim = alr_sequence.shape
        p = int(self.ar_coefficients.size(0))
        q = int(self.ma_coefficients.size(0))
        memory = max(p, q)

        eta = torch.zeros((length, alr_dim), dtype=alr_sequence.dtype, device=alr_sequence.device)
        residual = torch.zeros_like(eta)
        if memory > 0:
            eta[:memory] = alr_sequence[:memory]

        for t in range(memory, length):
            eta_t = torch.zeros(alr_dim, dtype=alr_sequence.dtype, device=alr_sequence.device)
            for lag in range(p):
                eta_t = eta_t + self.ar_coefficients[lag].to(dtype=alr_sequence.dtype, device=alr_sequence.device) @ alr_sequence[t - lag - 1]
            for lag in range(q):
                eta_t = eta_t + self.ma_coefficients[lag].to(dtype=alr_sequence.dtype, device=alr_sequence.device) @ (
                    alr_sequence[t - lag - 1] - eta[t - lag - 1]
                )
            eta[t] = eta_t
            residual[t] = alr_sequence[t] - eta_t

        return eta, residual

    def _predict_sequence(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        pred = probs.clone()
        p = int(self.ar_coefficients.size(0))
        q = int(self.ma_coefficients.size(0))
        memory = max(p, q)
        start = max(memory - 1, 0)
        if probs.size(1) <= start:
            return pred

        positive = _strictly_positive_simplex(probs, self.cfg.simplex_eps)
        for batch_idx in range(probs.size(0)):
            alr_sequence = _simplex_to_alr(positive[batch_idx], self.reference_index, self.cfg.simplex_eps)
            _, residual = self._compute_observed_state(alr_sequence)

            for step_idx in range(start, int(alr_sequence.size(0))):
                eta_next = torch.zeros(alr_sequence.size(-1), dtype=alr_sequence.dtype, device=alr_sequence.device)
                for lag in range(p):
                    eta_next = eta_next + self.ar_coefficients[lag].to(dtype=alr_sequence.dtype, device=alr_sequence.device) @ alr_sequence[step_idx - lag]
                for lag in range(q):
                    eta_next = eta_next + self.ma_coefficients[lag].to(dtype=alr_sequence.dtype, device=alr_sequence.device) @ residual[step_idx - lag]
                pred[batch_idx, step_idx] = _alr_to_simplex(
                    eta_next,
                    self.reference_index,
                    self.cfg.prob_dim,
                    self.cfg.simplex_eps,
                )
        return pred

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
            raise ValueError("runtime ablations are not supported for MODEL=bdarma")

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
