from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Callable, Dict

import torch.nn as nn

from .analog_successor import AnalogSuccessorBaseline, AnalogSuccessorConfig
from .bdarma import BDARMABaseline as BDARMAMAPBaseline, BDARMAConfig as BDARMAMAPConfig
from .bdarma_stan import BDARMABaseline, BDARMAConfig
from .cast_core import CASTConfig, CASTModel, apply_named_ablation, freeze_static_ablation_params
from .compositional_ets import CompositionalETSBaseline, CompositionalETSConfig
from .copy_baselines import CopyBaseline, CopyBaselineConfig
from .ilr_var import ILRVARBaseline, ILRVARConfig
from .naive_transformer import NaiveTransformerBaseline, NaiveTransformerConfig
from .nhits_simplex import NHitsSimplexBaseline, NHitsSimplexConfig
from .tslib_simplex import TSLibSimplexBaseline, TSLibSimplexConfig

AVAILABLE_MODELS: Dict[str, str] = {
    "cast": "Causal Anchored Simplex Transport",
    "persistence": "Simple persistence baseline that predicts the next simplex vector as the current one",
    "seasonal_naive": "Simple seasonal-naive baseline that copies the simplex vector from one seasonal period ago",
    "analog_successor": "Non-neural kNN analog successor baseline fit on train-split windows",
    "bdarma": "Canonical single-series Stan B-DARMA baseline with informative priors",
    "bdarma_map": "Approximate pooled MAP B-DARMA-style baseline in ALR coordinates",
    "compositional_ets": "Classical compositional ETS/state-space baseline in ilr coordinates",
    "ilr_var": "Classical ilr-space vector autoregression baseline fit on the train split",
    "transformer_naive": "Vanilla decoder-only Transformer baseline with simplex output projection",
    "dlinear": "Time-Series-Library DLinear forecasting baseline with a simplex head",
    "timemixer": "TimeMixer-style multiscale decomposition forecasting baseline with a simplex head",
    "itransformer": "Time-Series-Library iTransformer forecasting baseline with a simplex head",
    "informer": "Time-Series-Library Informer-style forecasting baseline with a simplex head",
    "autoformer": "Time-Series-Library Autoformer-style forecasting baseline with a simplex head",
    "tide": "Time-Series-Library TiDE-style forecasting baseline with a simplex head",
    "tcn": "Temporal convolutional forecasting baseline with a simplex head",
    "gru": "GRU forecasting baseline with a simplex head",
    "lstm": "LSTM forecasting baseline with a simplex head",
    "nhits": "N-HiTS-style residual MLP forecasting baseline with a simplex head",
}


@dataclass
class ModelBuildResult:
    name: str
    model: nn.Module
    config: object
    run_label: str
    freeze_for_training: Callable[[], None]
    requires_training: bool = True


def _first_non_empty_env(*names: str) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _env_bool(name: str, default: bool, *fallback_names: str) -> bool:
    raw = _first_non_empty_env(name, *fallback_names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *fallback_names: str) -> int:
    raw = _first_non_empty_env(name, *fallback_names)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float, *fallback_names: str) -> float:
    raw = _first_non_empty_env(name, *fallback_names)
    if raw is None:
        return default
    return float(raw)


def _env_float_tuple(name: str, default: tuple[float, ...], *fallback_names: str) -> tuple[float, ...]:
    raw = _first_non_empty_env(name, *fallback_names)
    if raw is None:
        return default
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one comma-separated float")
    return values


def _env_int_tuple(name: str, default: tuple[int, ...], *fallback_names: str) -> tuple[int, ...]:
    raw = _first_non_empty_env(name, *fallback_names)
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one comma-separated integer")
    return values


def _require_full_ablation(train_config, model_name: str) -> None:
    if train_config.ablation != "full":
        raise ValueError(f"MODEL={model_name} supports ABLATION=full only.")


def _default_seasonal_period(task) -> int:
    name = task.name.strip().lower()
    if name == "ember_monthly":
        return 12
    if name == "cdc_weekly_deaths":
        return 52
    return 1


def _build_cast(train_config, task) -> ModelBuildResult:
    overrides = dict(task.model_config_overrides(train_config, model_name="cast"))
    if "n_simplex_head" in overrides and "n_retrieval_head" not in overrides:
        overrides["n_retrieval_head"] = overrides.pop("n_simplex_head")
    if "simplex_head_dim" in overrides and "retrieval_head_dim" not in overrides:
        overrides["retrieval_head_dim"] = overrides.pop("simplex_head_dim")

    ordered_support = task.ordered_probability_support()
    has_ordered_support = ordered_support is not None

    cfg_kwargs = {
        "prob_dim": task.prob_dim,
        "max_seq_len": train_config.model_max_seq_len,
        "d_model": train_config.d_model,
        "n_layer": train_config.n_layer,
        "n_head": train_config.n_head,
        "d_ff": train_config.d_ff if train_config.d_ff is not None else 4 * int(train_config.d_model),
        "dropout": train_config.dropout,
        "attn_dropout": train_config.attn_dropout,
        "resid_dropout": train_config.resid_dropout,
        "normalize_input": train_config.normalize_input,
        "qk_norm": train_config.qk_norm,
        "loss_type": train_config.loss_type,
        "aux_prior_loss_weight": 0.0,
        "transport_reg_weight": 0.0,
        "tangent_reg_weight": 0.0,
        "struct_reg_weight": _env_float("CAST_STRUCT_REG_WEIGHT", 0.0),
        "struct_smooth_weight": _env_float("CAST_STRUCT_SMOOTH_WEIGHT", 0.0),
        "use_log_stem": _env_bool("CAST_USE_LOG_STEM", _env_bool("USE_LOG_STEM", True)),
        "use_input_stats": (
            True if train_config.use_input_stats is None else bool(train_config.use_input_stats)
        ),
        "use_ordered_support": _env_bool("CAST_USE_ORDERED_SUPPORT", has_ordered_support),
        "use_successor_retrieval": _env_bool("CAST_USE_SUCCESSOR_RETRIEVAL", True),
        "use_persistence_mix": _env_bool("CAST_USE_PERSISTENCE_MIX", True),
        "use_current_anchor_control": False,
        "use_tangent_flow": False,
        "use_local_transport": _env_bool("CAST_USE_LOCAL_TRANSPORT", has_ordered_support),
        "use_dense_transport_control": False,
        "use_support_dependent_kernel": _env_bool("CAST_USE_SUPPORT_DEPENDENT_KERNEL", True),
        "n_retrieval_head": (
            int(train_config.n_simplex_head)
            if train_config.n_simplex_head is not None
            else _env_int("CAST_N_RETRIEVAL_HEAD", 2, "N_SIMPLEX_HEAD")
        ),
        "retrieval_head_dim": (
            int(train_config.simplex_head_dim)
            if train_config.simplex_head_dim is not None
            else _env_int("CAST_RETRIEVAL_HEAD_DIM", 64, "SIMPLEX_HEAD_DIM")
        ),
        "retrieval_attn_dropout": _env_float(
            "CAST_RETRIEVAL_ATTN_DROPOUT",
            0.0,
            "RETRIEVAL_ATTN_DROPOUT",
        ),
        "anchor_persistence_init": _env_float("CAST_ANCHOR_PERSISTENCE_INIT", 0.55),
        "anchor_lambda_min": _env_float("CAST_ANCHOR_LAMBDA_MIN", 0.05),
        "anchor_lambda_max": _env_float("CAST_ANCHOR_LAMBDA_MAX", 0.95),
        "tangent_basis_dim": _env_int("CAST_TANGENT_BASIS_DIM", 8),
        "tangent_max_amplitude": _env_float("CAST_TANGENT_MAX_AMPLITUDE", 0.75),
        "tangent_amplitude_init": _env_float("CAST_TANGENT_AMPLITUDE_INIT", 0.05),
        "local_transport_radius": _env_int("CAST_LOCAL_TRANSPORT_RADIUS", 1),
        "local_transport_basis_dim": _env_int("CAST_LOCAL_TRANSPORT_BASIS_DIM", 2),
        "local_transport_identity_bias": _env_float("CAST_LOCAL_TRANSPORT_IDENTITY_BIAS", 5.0),
        "local_transport_strength_init": _env_float("CAST_LOCAL_TRANSPORT_STRENGTH_INIT", 0.02),
        "local_transport_max_strength": _env_float("CAST_LOCAL_TRANSPORT_MAX_STRENGTH", 0.20),
        "local_transport_max_mean_shift": _env_float("CAST_LOCAL_TRANSPORT_MAX_MEAN_SHIFT", 0.25),
        "local_transport_std_shift_fraction": _env_float(
            "CAST_LOCAL_TRANSPORT_STD_SHIFT_FRACTION",
            0.10,
        ),
        "dense_transport_rho_init": _env_float(
            "CAST_DENSE_TRANSPORT_RHO_INIT",
            train_config.transport_rho_init,
            "TRANSPORT_RHO_INIT",
        ),
    }
    cfg_kwargs.update(overrides)

    cfg = apply_named_ablation(CASTConfig(**cfg_kwargs), train_config.ablation)
    if not cfg.use_ordered_support:
        cfg = replace(
            cfg,
            use_ordered_support=False,
            use_local_transport=False,
            use_support_dependent_kernel=False,
        )
    model = CASTModel(cfg)
    run_label = "cast" if train_config.ablation == "full" else f"cast/{train_config.ablation}"
    return ModelBuildResult(
        name="cast",
        model=model,
        config=cfg,
        run_label=run_label,
        freeze_for_training=lambda: freeze_static_ablation_params(model, cfg),
    )


def _build_transformer_naive(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "transformer_naive")
    overrides = task.model_config_overrides(train_config, model_name="transformer_naive")
    matched_d_model = _env_int("NAIVE_D_MODEL", int(train_config.d_model))
    cfg_kwargs = {
        "prob_dim": task.prob_dim,
        "max_seq_len": train_config.model_max_seq_len,
        "d_model": matched_d_model,
        "n_layer": train_config.n_layer,
        "n_head": train_config.n_head,
        "dropout": train_config.dropout,
        "loss_type": train_config.loss_type,
    }
    cfg_kwargs.update(overrides)
    cfg = NaiveTransformerConfig(**cfg_kwargs)
    model = NaiveTransformerBaseline(cfg)
    return ModelBuildResult(
        name="transformer_naive",
        model=model,
        config=cfg,
        run_label="transformer_naive",
        freeze_for_training=lambda: None,
    )


def _build_copy_baseline(train_config, task, *, kind: str) -> ModelBuildResult:
    _require_full_ablation(train_config, kind)
    kind = kind.strip().lower()
    if kind == "persistence":
        cfg = CopyBaselineConfig(
            kind="persistence",
            prob_dim=task.prob_dim,
            forecast_lag=1,
            simplex_eps=float(os.environ.get("COPY_BASELINE_SIMPLEX_EPS", "1e-8")),
            loss_type=train_config.loss_type,
        )
        run_label = "persistence"
    elif kind == "seasonal_naive":
        default_period = _default_seasonal_period(task)
        period = int(
            os.environ.get(
                "SEASONAL_NAIVE_PERIOD",
                os.environ.get("SEASONAL_PERIOD", str(default_period)),
            )
        )
        cfg = CopyBaselineConfig(
            kind="seasonal_naive",
            prob_dim=task.prob_dim,
            forecast_lag=period,
            simplex_eps=float(os.environ.get("COPY_BASELINE_SIMPLEX_EPS", "1e-8")),
            loss_type=train_config.loss_type,
        )
        run_label = f"seasonal_naive/s{cfg.forecast_lag}"
    else:
        raise ValueError(f"unsupported copy baseline kind={kind!r}")

    model = CopyBaseline(cfg)
    return ModelBuildResult(
        name=kind,
        model=model,
        config=cfg,
        run_label=run_label,
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def _build_tslib_simplex(train_config, task, *, backbone_name: str) -> ModelBuildResult:
    _require_full_ablation(train_config, backbone_name)
    overrides = task.model_config_overrides(train_config, model_name=backbone_name)
    max_seq_len = int(train_config.model_max_seq_len)
    moving_avg = int(
        os.environ.get(
            f"{backbone_name.upper()}_MOVING_AVG",
            os.environ.get("DLINEAR_MOVING_AVG", os.environ.get("TSLIB_MOVING_AVG", "25")),
        )
    )
    default_window_batch_size = "32" if backbone_name == "autoformer" else "256"
    cfg_kwargs = {
        "backbone": backbone_name,
        "prob_dim": task.prob_dim,
        "max_seq_len": max_seq_len,
        "d_model": train_config.d_model,
        "n_layer": train_config.n_layer,
        "n_head": train_config.n_head,
        "d_ff": train_config.d_ff if train_config.d_ff is not None else 4 * int(train_config.d_model),
        "dropout": train_config.dropout,
        "simplex_eps": float(os.environ.get("TSLIB_SIMPLEX_EPS", "1e-8")),
        "loss_type": train_config.loss_type,
        "normalize_input": _env_bool("TSLIB_NORMALIZE_INPUT", train_config.normalize_input),
        "moving_avg": moving_avg,
        "individual": _env_bool("DLINEAR_INDIVIDUAL", _env_bool("TSLIB_INDIVIDUAL", False)),
        "window_batch_size": int(
            os.environ.get(
                f"{backbone_name.upper()}_WINDOW_BATCH_SIZE",
                os.environ.get("TSLIB_WINDOW_BATCH_SIZE", default_window_batch_size),
            )
        ),
        "tcn_kernel_size": int(os.environ.get("TCN_KERNEL_SIZE", os.environ.get("TSLIB_TCN_KERNEL_SIZE", "3"))),
        "tcn_dilation_base": int(
            os.environ.get("TCN_DILATION_BASE", os.environ.get("TSLIB_TCN_DILATION_BASE", "2"))
        ),
        "decoder_layers": int(
            os.environ.get(
                f"{backbone_name.upper()}_DECODER_LAYERS",
                os.environ.get("TSLIB_DECODER_LAYERS", "1"),
            )
        ),
        "attention_factor": int(
            os.environ.get(
                f"{backbone_name.upper()}_FACTOR",
                os.environ.get("TSLIB_ATTENTION_FACTOR", "5"),
            )
        ),
        "informer_distil": _env_bool("INFORMER_DISTIL", _env_bool("TSLIB_INFORMER_DISTIL", True)),
        "tide_feature_dim": int(os.environ.get("TIDE_FEATURE_DIM", os.environ.get("TSLIB_TIDE_FEATURE_DIM", "2"))),
        "timemixer_levels": int(os.environ.get("TIMEMIXER_LEVELS", os.environ.get("TSLIB_TIMEMIXER_LEVELS", "3"))),
        "timemixer_downsample_factor": int(
            os.environ.get("TIMEMIXER_DOWNSAMPLE_FACTOR", os.environ.get("TSLIB_TIMEMIXER_DOWNSAMPLE_FACTOR", "2"))
        ),
        "timemixer_channel_mixing": _env_bool(
            "TIMEMIXER_CHANNEL_MIXING",
            _env_bool("TSLIB_TIMEMIXER_CHANNEL_MIXING", True),
        ),
    }
    cfg_kwargs.update(overrides)
    cfg = TSLibSimplexConfig(**cfg_kwargs)
    model = TSLibSimplexBaseline(cfg)
    return ModelBuildResult(
        name=backbone_name,
        model=model,
        config=cfg,
        run_label=backbone_name,
        freeze_for_training=lambda: None,
    )


def _build_nhits(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "nhits")
    overrides = task.model_config_overrides(train_config, model_name="nhits")
    cfg_kwargs = {
        "prob_dim": task.prob_dim,
        "max_seq_len": int(train_config.model_max_seq_len),
        "coordinate_mode": os.environ.get("NHITS_COORDINATE_MODE", "raw_clr").strip().lower() or "raw_clr",
        "num_stacks": _env_int("NHITS_NUM_STACKS", 3),
        "blocks_per_stack": _env_int("NHITS_BLOCKS_PER_STACK", 2),
        "hidden_size": _env_int("NHITS_HIDDEN_SIZE", int(train_config.d_model), "D_MODEL", "D_K"),
        "num_layers": _env_int("NHITS_NUM_LAYERS", 2),
        "pool_kernels": _env_int_tuple("NHITS_POOL_KERNELS", (1, 2, 4)),
        "dropout": _env_float("NHITS_DROPOUT", float(train_config.dropout), "DROPOUT"),
        "simplex_eps": float(os.environ.get("NHITS_SIMPLEX_EPS", "1e-8")),
        "window_batch_size": _env_int("NHITS_WINDOW_BATCH_SIZE", 256),
        "loss_type": train_config.loss_type,
    }
    cfg_kwargs.update(overrides)
    cfg = NHitsSimplexConfig(**cfg_kwargs)
    model = NHitsSimplexBaseline(cfg)
    return ModelBuildResult(
        name="nhits",
        model=model,
        config=cfg,
        run_label=f"nhits/{cfg.coordinate_mode}_s{cfg.num_stacks}x{cfg.blocks_per_stack}_h{cfg.hidden_size}",
        freeze_for_training=lambda: None,
    )


def _build_ilr_var(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "ilr_var")
    fit_split = os.environ.get("ILR_VAR_FIT_SPLIT", "train").strip().lower() or "train"
    order_raw = os.environ.get("ILR_VAR_ORDER")
    order_override = None if order_raw is None or not order_raw.strip() else int(order_raw)
    ilr_dim = max(int(task.prob_dim) - 1, 1)
    max_features = int(os.environ.get("ILR_VAR_MAX_FEATURES", "4096"))
    feature_budget_lags = max(1, max_features // ilr_dim)
    default_max_lags = max(1, min(int(train_config.model_max_seq_len), 8, feature_budget_lags))
    max_lags = int(os.environ.get("ILR_VAR_MAX_LAGS", default_max_lags))
    order_selection = (
        "fixed"
        if order_override is not None
        else os.environ.get("ILR_VAR_ORDER_SELECTION", "aic").strip().lower()
    )
    cfg = ILRVARConfig(
        prob_dim=task.prob_dim,
        lag_order=1 if order_override is None else order_override,
        max_lags=max_lags,
        order_selection=order_selection,
        ridge=float(os.environ.get("ILR_VAR_RIDGE", "1e-4")),
        simplex_eps=float(os.environ.get("ILR_VAR_SIMPLEX_EPS", "1e-8")),
        loss_type=train_config.loss_type,
        fit_split=fit_split,
        fit_chunk_size=int(os.environ.get("ILR_VAR_FIT_CHUNK_SIZE", "1024")),
    )
    model = ILRVARBaseline.fit_from_task(cfg, task)
    order_label = f"p{model.cfg.effective_order}" if model.cfg.effective_order > 0 else "persistence_fallback"
    return ModelBuildResult(
        name="ilr_var",
        model=model,
        config=model.cfg,
        run_label=f"ilr_var/{order_label}",
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def _build_compositional_ets(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "compositional_ets")
    fit_split = os.environ.get("COMPOSITIONAL_ETS_FIT_SPLIT", "train").strip().lower() or "train"
    default_period = _default_seasonal_period(task)
    seasonal_period = int(os.environ.get("COMPOSITIONAL_ETS_SEASONAL_PERIOD", str(default_period)) or default_period)
    cfg = CompositionalETSConfig(
        prob_dim=task.prob_dim,
        mode=os.environ.get("COMPOSITIONAL_ETS_MODE", "auto").strip().lower() or "auto",
        seasonal_period=seasonal_period,
        alpha_grid=_env_float_tuple("COMPOSITIONAL_ETS_ALPHA_GRID", (0.1, 0.2, 0.4, 0.6, 0.8)),
        beta_grid=_env_float_tuple("COMPOSITIONAL_ETS_BETA_GRID", (0.0, 0.05, 0.1, 0.2)),
        gamma_grid=_env_float_tuple("COMPOSITIONAL_ETS_GAMMA_GRID", (0.0, 0.05, 0.1, 0.2)),
        phi_grid=_env_float_tuple("COMPOSITIONAL_ETS_PHI_GRID", (0.8, 0.9, 0.98, 1.0)),
        simplex_eps=float(os.environ.get("COMPOSITIONAL_ETS_SIMPLEX_EPS", "1e-8")),
        loss_type=train_config.loss_type,
        fit_split=fit_split,
        selection_metric=os.environ.get("COMPOSITIONAL_ETS_SELECTION_METRIC", "kl").strip().lower() or "kl",
    )
    model = CompositionalETSBaseline.fit_from_task(cfg, task)
    if model.cfg.fitted_num_samples <= 0:
        mode_label = "persistence_fallback"
    elif model.cfg.effective_mode == "seasonal":
        mode_label = f"seasonal{model.cfg.seasonal_period}"
    else:
        mode_label = model.cfg.effective_mode
    return ModelBuildResult(
        name="compositional_ets",
        model=model,
        config=model.cfg,
        run_label=f"compositional_ets/{mode_label}",
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def _build_analog_successor(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "analog_successor")
    cfg = AnalogSuccessorConfig(
        prob_dim=task.prob_dim,
        window=_env_int("ANALOG_WINDOW", 8),
        stride=_env_int("ANALOG_STRIDE", 1),
        k=_env_int("ANALOG_K", 16),
        coordinate_mode=os.environ.get("ANALOG_COORDINATE_MODE", "ilr").strip().lower() or "ilr",
        include_deltas=_env_bool("ANALOG_INCLUDE_DELTAS", True),
        value_mode=os.environ.get("ANALOG_VALUE_MODE", "delta").strip().lower() or "delta",
        blend_weight=_env_float("ANALOG_BLEND_WEIGHT", 0.5),
        distance=os.environ.get("ANALOG_DISTANCE", "euclidean").strip().lower() or "euclidean",
        temperature=_env_float("ANALOG_TEMPERATURE", 0.1),
        max_memory_items=_env_int("ANALOG_MAX_MEMORY_ITEMS", 200000),
        memory_split=os.environ.get("ANALOG_MEMORY_SPLIT", "train").strip().lower() or "train",
        memory_subsample_seed=_env_int("ANALOG_MEMORY_SUBSAMPLE_SEED", train_config.seed),
        query_chunk_size=_env_int("ANALOG_QUERY_CHUNK_SIZE", 256),
        memory_chunk_size=_env_int("ANALOG_MEMORY_CHUNK_SIZE", 16384),
        simplex_eps=float(os.environ.get("ANALOG_SIMPLEX_EPS", "1e-8")),
        loss_type=train_config.loss_type,
    )
    model = AnalogSuccessorBaseline.fit_from_task(cfg, task)
    stride_label = "" if model.cfg.stride == 1 else f"_s{model.cfg.stride}"
    mode_label = f"w{model.cfg.window}{stride_label}_k{model.cfg.k}_{model.cfg.value_mode}"
    if model.cfg.effective_memory_items <= 0:
        mode_label = f"{mode_label}_persistence_fallback"
    return ModelBuildResult(
        name="analog_successor",
        model=model,
        config=model.cfg,
        run_label=f"analog_successor/{mode_label}",
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def _build_bdarma(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "bdarma")
    fit_split = os.environ.get("BDARMA_FIT_SPLIT", "train").strip().lower() or "train"
    cfg = BDARMAConfig(
        prob_dim=task.prob_dim,
        ar_order=int(os.environ.get("BDARMA_P", "1")),
        ma_order=int(os.environ.get("BDARMA_Q", "0")),
        loss_type=train_config.loss_type,
        fit_split=fit_split,
        simplex_eps=float(os.environ.get("BDARMA_SIMPLEX_EPS", "1e-6")),
        prior=os.environ.get("BDARMA_PRIOR", "informative").strip().lower() or "informative",
        chains=int(os.environ.get("BDARMA_CHAINS", "4")),
        parallel_chains=int(os.environ.get("BDARMA_PARALLEL_CHAINS", os.environ.get("BDARMA_CHAINS", "4"))),
        iter_warmup=int(os.environ.get("BDARMA_ITER_WARMUP", "1000")),
        iter_sampling=int(os.environ.get("BDARMA_ITER_SAMPLING", "1000")),
        adapt_delta=float(os.environ.get("BDARMA_ADAPT_DELTA", "0.9")),
        max_treedepth=int(os.environ.get("BDARMA_MAX_TREEDEPTH", "11")),
        init=float(os.environ.get("BDARMA_INIT", "0.3")),
        seed=train_config.seed,
        cache_dir=os.environ.get("BDARMA_CACHE_DIR", os.path.join("outputs", "bdarma_cache")),
        fit_timeout_sec=int(os.environ.get("BDARMA_FIT_TIMEOUT_SEC", "86400")),
    )
    model = BDARMABaseline.fit_from_task(cfg, task)
    order_label = (
        "persistence_fallback"
        if model.cfg.fitted_num_samples <= 0
        else f"p{model.cfg.effective_ar_order}q{model.cfg.effective_ma_order}"
    )
    return ModelBuildResult(
        name="bdarma",
        model=model,
        config=model.cfg,
        run_label=f"bdarma/{order_label}",
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def _build_bdarma_map(train_config, task) -> ModelBuildResult:
    _require_full_ablation(train_config, "bdarma_map")
    fit_split = os.environ.get("BDARMA_MAP_FIT_SPLIT", os.environ.get("BDARMA_FIT_SPLIT", "train")).strip().lower() or "train"
    ar_raw = os.environ.get("BDARMA_MAP_P", os.environ.get("BDARMA_P"))
    ar_locked = ar_raw is not None and bool(ar_raw.strip())
    ar_order = 1 if not ar_locked else int(ar_raw)
    ma_raw = os.environ.get("BDARMA_MAP_Q", os.environ.get("BDARMA_Q"))
    ma_locked = ma_raw is not None and bool(ma_raw.strip())
    ma_order = 0 if not ma_locked else int(ma_raw)
    max_ar_default = max(1, min(int(train_config.model_max_seq_len), 4))
    max_ar_order = ar_order if ar_locked else int(os.environ.get("BDARMA_MAP_MAX_P", os.environ.get("BDARMA_MAX_P", str(max_ar_default))))
    max_ma_order = ma_order if ma_locked else int(os.environ.get("BDARMA_MAP_MAX_Q", os.environ.get("BDARMA_MAX_Q", "0")))
    order_selection = (
        "fixed"
        if ar_locked and ma_locked
        else os.environ.get("BDARMA_MAP_ORDER_SELECTION", os.environ.get("BDARMA_ORDER_SELECTION", "aic")).strip().lower()
    )
    cfg = BDARMAMAPConfig(
        prob_dim=task.prob_dim,
        ar_order=ar_order,
        ma_order=ma_order,
        max_ar_order=max_ar_order,
        max_ma_order=max_ma_order,
        ar_order_locked=ar_locked,
        ma_order_locked=ma_locked,
        order_selection=order_selection,
        ridge=float(os.environ.get("BDARMA_MAP_RIDGE", os.environ.get("BDARMA_RIDGE", "1e-3"))),
        simplex_eps=float(os.environ.get("BDARMA_MAP_SIMPLEX_EPS", os.environ.get("BDARMA_SIMPLEX_EPS", "1e-8"))),
        dirichlet_eps=float(os.environ.get("BDARMA_MAP_DIRICHLET_EPS", os.environ.get("BDARMA_DIRICHLET_EPS", "1e-6"))),
        loss_type=train_config.loss_type,
        fit_split=fit_split,
        fit_max_steps=int(os.environ.get("BDARMA_MAP_FIT_MAX_STEPS", os.environ.get("BDARMA_FIT_MAX_STEPS", "400"))),
        fit_eval_interval=int(os.environ.get("BDARMA_MAP_FIT_EVAL_INTERVAL", os.environ.get("BDARMA_FIT_EVAL_INTERVAL", "10"))),
        fit_patience=int(os.environ.get("BDARMA_MAP_FIT_PATIENCE", os.environ.get("BDARMA_FIT_PATIENCE", "80"))),
        fit_sequence_batch_size=int(
            os.environ.get("BDARMA_MAP_FIT_SEQUENCE_BATCH_SIZE", os.environ.get("BDARMA_FIT_SEQUENCE_BATCH_SIZE", "16"))
        ),
        fit_lr=float(os.environ.get("BDARMA_MAP_FIT_LR", os.environ.get("BDARMA_FIT_LR", "5e-2"))),
        fit_grad_clip=float(os.environ.get("BDARMA_MAP_FIT_GRAD_CLIP", os.environ.get("BDARMA_FIT_GRAD_CLIP", "10.0"))),
        coef_bound=float(os.environ.get("BDARMA_MAP_COEF_BOUND", os.environ.get("BDARMA_COEF_BOUND", "0.95"))),
        init_concentration=float(
            os.environ.get("BDARMA_MAP_INIT_CONCENTRATION", os.environ.get("BDARMA_INIT_CONCENTRATION", "50.0"))
        ),
        min_concentration=float(
            os.environ.get("BDARMA_MAP_MIN_CONCENTRATION", os.environ.get("BDARMA_MIN_CONCENTRATION", "1e-2"))
        ),
        prior_scale_mean=float(
            os.environ.get("BDARMA_MAP_PRIOR_SCALE_MEAN", os.environ.get("BDARMA_PRIOR_SCALE_MEAN", "2.0"))
        ),
        prior_scale_ar=float(
            os.environ.get("BDARMA_MAP_PRIOR_SCALE_AR", os.environ.get("BDARMA_PRIOR_SCALE_AR", "0.5"))
        ),
        prior_scale_ma=float(
            os.environ.get("BDARMA_MAP_PRIOR_SCALE_MA", os.environ.get("BDARMA_PRIOR_SCALE_MA", "0.5"))
        ),
        prior_scale_log_concentration=float(
            os.environ.get(
                "BDARMA_MAP_PRIOR_SCALE_LOG_CONCENTRATION",
                os.environ.get("BDARMA_PRIOR_SCALE_LOG_CONCENTRATION", "2.0"),
            )
        ),
    )
    model = BDARMAMAPBaseline.fit_from_task(cfg, task)
    order_label = (
        "persistence_fallback"
        if model.cfg.fitted_num_samples <= 0
        else f"p{model.cfg.effective_ar_order}q{model.cfg.effective_ma_order}"
    )
    return ModelBuildResult(
        name="bdarma_map",
        model=model,
        config=model.cfg,
        run_label=f"bdarma_map/{order_label}",
        freeze_for_training=lambda: None,
        requires_training=False,
    )


def build_model(train_config, task) -> ModelBuildResult:
    model_name = train_config.model_name.strip().lower()
    if model_name in {"cast", "causal_anchored_simplex_transport"}:
        return _build_cast(train_config, task)
    if model_name in {"persistence", "persist"}:
        return _build_copy_baseline(train_config, task, kind="persistence")
    if model_name in {"seasonal_naive", "seasonal-naive", "seasonality", "seasonal"}:
        return _build_copy_baseline(train_config, task, kind="seasonal_naive")
    if model_name in {"analog_successor", "analog-successor", "analog", "knn_successor", "knn-successor"}:
        return _build_analog_successor(train_config, task)
    if model_name in {"bdarma", "b-darma"}:
        return _build_bdarma(train_config, task)
    if model_name in {"bdarma_map", "bdarma-map", "bdarma_approx"}:
        return _build_bdarma_map(train_config, task)
    if model_name in {"ilr_var", "ilr-var", "var_ilr"}:
        return _build_ilr_var(train_config, task)
    if model_name in {"compositional_ets", "compositional-ets", "ets", "ilr_ets"}:
        return _build_compositional_ets(train_config, task)
    if model_name in {"transformer_naive", "naive_transformer", "transformer", "vanilla_transformer"}:
        return _build_transformer_naive(train_config, task)
    if model_name in {"nhits", "n_hits", "nhits_simplex", "n-hits"}:
        return _build_nhits(train_config, task)
    if model_name in {"dlinear", "dlinear_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="dlinear")
    if model_name in {"timemixer", "time_mixer", "timemixer_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="timemixer")
    if model_name in {"itransformer", "i_transformer", "itransformer_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="itransformer")
    if model_name in {"informer", "informer_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="informer")
    if model_name in {"autoformer", "autoformer_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="autoformer")
    if model_name in {"tide", "tide_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="tide")
    if model_name in {"tcn", "tcn_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="tcn")
    if model_name in {"gru", "gru_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="gru")
    if model_name in {"lstm", "lstm_simplex"}:
        return _build_tslib_simplex(train_config, task, backbone_name="lstm")

    available = ", ".join(sorted(AVAILABLE_MODELS))
    raise ValueError(f"unknown MODEL={train_config.model_name!r}. Available models: {available}")
