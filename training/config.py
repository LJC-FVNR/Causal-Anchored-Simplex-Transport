from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from training.dataset_defaults import default_block_size

ALL_ABLATIONS = [
    "full",
    "without_structural_reg",
    "anchor_only",
    "single_head_retrieval",
    "fixed_local_kernel",
    "without_persistence_mix",
]


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    return None if raw is None or raw.strip() == "" else int(raw)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_optional_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    return None if raw is None or raw.strip() == "" else float(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    dataset_name: str
    batch_size: int
    block_size: int
    model_max_seq_len: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_iters: int
    log_interval: int
    grad_accum_steps: int
    warmup_iters: int
    use_amp: bool
    use_compile: bool
    weight_decay: float
    clip_grad: float
    seed: int
    sample_new_tokens: int
    sample_temperature: float
    sample_prompt: str
    ablation: str
    run_tag: str
    d_model: int
    n_layer: int
    n_head: int
    d_ff: Optional[int]
    dropout: float
    attn_dropout: float
    resid_dropout: float
    normalize_input: bool
    qk_norm: bool
    loss_type: str
    aux_prior_loss_weight: float
    transport_reg_weight: float
    transport_rho_init: float
    prior_blend_init: float
    retrieval_mix_init: float
    tau_init: float
    tau_min: float
    use_input_stats: Optional[bool]
    n_simplex_head: Optional[int]
    simplex_head_dim: Optional[int]
    innovation_hidden_mult: Optional[float]
    save_best_checkpoint: bool
    checkpoint_dir: str
    run_offline_eval: bool
    offline_eval_batch_size: int
    save_eval_artifact: bool
    eval_artifact_dir: str
    run_rollout_eval: bool
    rollout_eval_batch_size: int
    rollout_max_examples: int
    rollout_context_steps: Optional[int]
    rollout_horizon_steps: Optional[int]
    save_rollout_artifact: bool
    rollout_artifact_dir: str
    skip_completed_identical: bool
    completed_search_root: str

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        d_model = env_optional_int("D_MODEL")
        if d_model is None:
            d_model = env_int("D_K", 512)

        model_name = env_str("MODEL", "cast").lower()
        dataset_name = env_str("DATASET", "tensor_dir").lower()
        seed = env_int("SEED", 42)
        ablation = env_str("ABLATION", "full").lower()
        if ablation not in ALL_ABLATIONS:
            raise ValueError(f"Unknown ABLATION={ablation}. Must be one of {ALL_ABLATIONS}")
        run_tag = env_str("RUN_TAG", f"{dataset_name}_{model_name}_{ablation}_seed{seed}")
        block_size = env_int("BLOCK_SIZE", default_block_size(dataset_name, fallback=1024))

        return cls(
            model_name=model_name,
            dataset_name=dataset_name,
            batch_size=env_int("BATCH_SIZE", 4),
            block_size=block_size,
            model_max_seq_len=env_int("MODEL_MAX_SEQ_LEN", block_size),
            learning_rate=env_float("LEARNING_RATE", 6e-4),
            max_iters=env_int("MAX_ITERS", 20000),
            eval_interval=env_int("EVAL_INTERVAL", 1000),
            eval_iters=env_int("EVAL_ITERS", 20),
            log_interval=env_int("LOG_INTERVAL", 50),
            grad_accum_steps=env_int("GRAD_ACCUM_STEPS", 16),
            warmup_iters=env_int("WARMUP_ITERS", 200),
            use_amp=env_bool("USE_AMP", True),
            use_compile=env_bool("USE_COMPILE", False),
            weight_decay=env_float("WEIGHT_DECAY", 1e-1),
            clip_grad=env_float("CLIP_GRAD", 1.0),
            seed=seed,
            sample_new_tokens=env_int("SAMPLE_NEW_TOKENS", 400),
            sample_temperature=env_float("SAMPLE_TEMPERATURE", 1.0),
            sample_prompt=env_str("SAMPLE_PROMPT", "ROMEO:"),
            ablation=ablation,
            run_tag=run_tag,
            d_model=d_model,
            n_layer=env_int("N_LAYER", 8),
            n_head=env_int("N_HEAD", 4),
            d_ff=env_optional_int("D_FF"),
            dropout=env_float("DROPOUT", 0.0),
            attn_dropout=env_float("ATTN_DROPOUT", 0.0),
            resid_dropout=env_float("RESID_DROPOUT", env_float("DROPOUT", 0.0)),
            normalize_input=env_bool("NORMALIZE_INPUT", True),
            qk_norm=env_bool("QK_NORM", False),
            loss_type=env_str("LOSS_TYPE", "ce"),
            aux_prior_loss_weight=env_float("AUX_PRIOR_LOSS_WEIGHT", 0.05),
            transport_reg_weight=env_float("TRANSPORT_REG_WEIGHT", 1e-4),
            transport_rho_init=env_float("TRANSPORT_RHO_INIT", 0.05),
            prior_blend_init=env_float("PRIOR_BLEND_INIT", 0.85),
            retrieval_mix_init=env_float("RETRIEVAL_MIX_INIT", 0.30),
            tau_init=env_float("TAU_INIT", 1.0),
            tau_min=env_float("TAU_MIN", 0.25),
            use_input_stats=env_optional_bool("USE_INPUT_STATS"),
            n_simplex_head=env_optional_int("N_SIMPLEX_HEAD"),
            simplex_head_dim=env_optional_int("SIMPLEX_HEAD_DIM"),
            innovation_hidden_mult=env_optional_float("INNOVATION_HIDDEN_MULT"),
            save_best_checkpoint=env_bool("SAVE_BEST_CHECKPOINT", True),
            checkpoint_dir=env_str("CHECKPOINT_DIR", "outputs/checkpoints"),
            run_offline_eval=env_bool("RUN_OFFLINE_EVAL", True),
            offline_eval_batch_size=env_int("OFFLINE_EVAL_BATCH_SIZE", env_int("BATCH_SIZE", 4)),
            save_eval_artifact=env_bool("SAVE_EVAL_ARTIFACT", True),
            eval_artifact_dir=env_str("EVAL_ARTIFACT_DIR", "outputs/eval_artifacts"),
            run_rollout_eval=env_bool("RUN_ROLLOUT_EVAL", True),
            rollout_eval_batch_size=env_int(
                "ROLLOUT_EVAL_BATCH_SIZE",
                env_int("OFFLINE_EVAL_BATCH_SIZE", env_int("BATCH_SIZE", 4)),
            ),
            rollout_max_examples=env_int("ROLLOUT_MAX_EXAMPLES", 8),
            rollout_context_steps=env_optional_int("ROLLOUT_CONTEXT_STEPS"),
            rollout_horizon_steps=env_optional_int("ROLLOUT_HORIZON_STEPS"),
            save_rollout_artifact=env_bool("SAVE_ROLLOUT_ARTIFACT", True),
            rollout_artifact_dir=env_str("ROLLOUT_ARTIFACT_DIR", "outputs/rollout_artifacts"),
            skip_completed_identical=env_bool("SKIP_COMPLETED_IDENTICAL", True),
            completed_search_root=env_str("COMPLETED_SEARCH_ROOT", os.environ.get("FULL_SWEEP_RUN_ROOT", "")),
        )

    def resolve_model_kwargs(
        self,
        *,
        prob_dim: int,
        task_overrides: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = dict(task_overrides or {})
        kwargs.update(
            {
                "prob_dim": prob_dim,
                "max_seq_len": self.model_max_seq_len,
                "d_model": self.d_model,
                "n_layer": self.n_layer,
                "n_head": self.n_head,
                "dropout": self.dropout,
                "attn_dropout": self.attn_dropout,
                "resid_dropout": self.resid_dropout,
                "normalize_input": self.normalize_input,
                "qk_norm": self.qk_norm,
                "loss_type": self.loss_type,
                "aux_prior_loss_weight": self.aux_prior_loss_weight,
                "transport_reg_weight": self.transport_reg_weight,
                "transport_rho_init": self.transport_rho_init,
                "prior_blend_init": self.prior_blend_init,
                "retrieval_mix_init": self.retrieval_mix_init,
                "tau_init": self.tau_init,
                "tau_min": self.tau_min,
            }
        )

        optional_overrides = {
            "d_ff": self.d_ff,
            "use_input_stats": self.use_input_stats,
            "n_simplex_head": self.n_simplex_head,
            "simplex_head_dim": self.simplex_head_dim,
            "innovation_hidden_mult": self.innovation_hidden_mult,
        }
        for key, value in optional_overrides.items():
            if value is not None:
                kwargs[key] = value

        return kwargs
