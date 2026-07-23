"""Find simulations that separate jump and latent-only GPLVM fits.

This diagnostic is intentionally simulation-only.  It gives every fitted model
the same random latent posterior, then grants every model an oracle post-fit
latent permutation, one fixed-posterior Adam M-step, and one diagnostic E-step.
The purpose is to identify data regimes where the explicit jump state remains
necessary even after this strongest-case alignment.

The script runs one configurable simulation at a time so cluster experiments
can iterate cheaply.  It saves numerical summaries, raw aligned results, and
state-specific latent and tuning plots.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from poor_man_gplvm import PoissonGPLVM1D, PoissonGPLVMJump1D


@dataclass(frozen=True)
class SimulationConfig:
    name: str = "candidate"
    n_neuron: int = 60
    n_time: int = 8_000
    n_latent_bin: int = 100
    movement_variance: float = 0.35
    tuning_lengthscale: float = 6.0
    p_move_to_jump: float = 0.06
    p_jump_to_move: float = 0.02
    rate_bias: float = -0.5
    simulation_seed: int = 103
    generative_param_seed: int = 123
    fitted_param_seed: int = 124
    posterior_seed: int = 5


@dataclass(frozen=True)
class RunConfig:
    em_iterations: int = 20
    m_step_step_size: float = 0.01
    m_step_maxiter: int = 1_000
    m_step_tol: float = 1e-10
    nojump_movement_scales: tuple[float, ...] = (0.35, 3.0, 10.0, 30.0)


@dataclass
class SimulationContext:
    config: SimulationConfig
    state: np.ndarray
    observations: np.ndarray
    tuning_true: np.ndarray
    log_posterior_init: np.ndarray


@dataclass
class FitResult:
    label: str
    model_kind: str
    movement_scale: float
    posterior: np.ndarray
    tuning: np.ndarray
    dynamics_posterior: np.ndarray | None
    fit_to_true: np.ndarray
    legacy_log_marginal: float
    posthoc_log_marginal: float
    posthoc_m_step_n_iter: int


def preflight(require_gpu: bool = True) -> dict:
    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, but JAX devices are {devices}")
    if require_gpu and shutil.which("ptxas") is None:
        raise RuntimeError("GPU required, but ptxas is not available")
    return {
        "hostname": platform.node(),
        "python": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "ptxas": shutil.which("ptxas"),
        "jax_devices": [str(device) for device in devices],
    }


def build_simulation(config: SimulationConfig) -> SimulationContext:
    generator = PoissonGPLVMJump1D(
        config.n_neuron,
        n_latent_bin=config.n_latent_bin,
        movement_variance=config.movement_variance,
        tuning_lengthscale=config.tuning_lengthscale,
        p_move_to_jump=config.p_move_to_jump,
        p_jump_to_move=config.p_jump_to_move,
        rng_init_int=config.generative_param_seed,
    )
    generative_params = generator.params.at[0].add(config.rate_bias)
    tuning_true = generator.get_tuning(
        generative_params, {}, generator.tuning_basis
    )
    state, observations = generator.sample(
        config.n_time,
        key=jax.random.PRNGKey(config.simulation_seed),
        tuning=tuning_true,
    )
    initial = jax.random.uniform(
        jax.random.PRNGKey(config.posterior_seed),
        shape=(config.n_time, config.n_latent_bin),
    )
    initial = initial / initial.sum(axis=1, keepdims=True)
    return SimulationContext(
        config=config,
        state=np.asarray(state),
        # This preserves integer count values while avoiding an old H100
        # compiler failure for the mixed float32-posterior @ int32-count
        # matrix product in get_statistics.
        observations=np.asarray(observations, dtype=np.float32),
        tuning_true=np.asarray(tuning_true),
        log_posterior_init=np.asarray(jnp.log(initial)),
    )


def _best_posterior_overlap_permutation(
    posterior: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    truth = state[:, 1].astype(int)
    n_latent = posterior.shape[1]
    overlap = np.zeros((n_latent, n_latent), dtype=np.float64)
    np.add.at(overlap, truth, posterior)
    true_index, fitted_index = linear_sum_assignment(-overlap)
    fit_to_true = np.empty(n_latent, dtype=int)
    fit_to_true[fitted_index] = true_index
    return fit_to_true


def _apply_permutation(
    posterior: np.ndarray,
    tuning: np.ndarray,
    fit_to_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_to_fit = np.argsort(fit_to_true)
    return posterior[:, true_to_fit], tuning[true_to_fit]


def _model(
    context: SimulationContext,
    model_kind: str,
    movement_scale: float,
):
    config = context.config
    kwargs = {
        "n_neuron": config.n_neuron,
        "n_latent_bin": config.n_latent_bin,
        "movement_variance": movement_scale,
        "tuning_lengthscale": config.tuning_lengthscale,
        "rng_init_int": config.fitted_param_seed,
    }
    if model_kind == "jump":
        return PoissonGPLVMJump1D(
            p_move_to_jump=config.p_move_to_jump,
            p_jump_to_move=config.p_jump_to_move,
            **kwargs,
        )
    if model_kind == "nojump":
        return PoissonGPLVM1D(**kwargs)
    raise ValueError(f"unknown model_kind={model_kind!r}")


def _posterior_from_fit(fit: dict, model_kind: str) -> np.ndarray:
    if model_kind == "jump":
        return np.asarray(fit["posterior_latent_marg"])
    return np.asarray(fit["posterior"])


def _posthoc_m_step_and_e_step(
    model,
    model_kind: str,
    observations: np.ndarray,
    aligned_posterior: np.ndarray,
    n_time_per_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, int]:
    posterior = aligned_posterior / aligned_posterior.sum(
        axis=1, keepdims=True
    )
    log_posterior = np.log(np.maximum(posterior, 1e-30))
    hyperparam = {
        "param_prior_std": model.param_prior_std,
        "smoothness_penalty": model.smoothness_penalty,
    }
    optimizer_state = model.opt_state_init_fun(model.params)
    m_step = model.m_step(
        model.params,
        observations,
        log_posterior,
        model.tuning_basis,
        hyperparam,
        opt_state_curr=optimizer_state,
    )
    tuning = np.asarray(
        model.get_tuning(
            m_step["params"], hyperparam, model.tuning_basis
        )
    )
    decoded = model.decode_latent(
        observations,
        tuning=tuning,
        n_time_per_chunk=n_time_per_chunk,
    )
    if model_kind == "jump":
        posterior_out = np.asarray(decoded["posterior_latent_marg"])
        dynamics = np.asarray(decoded["posterior_dynamics_marg"])
    else:
        posterior_out = np.asarray(decoded["posterior_all"])
        dynamics = None
    return (
        posterior_out,
        tuning,
        dynamics,
        float(decoded["log_marginal_final"]),
        int(m_step["n_iter"]),
    )


def fit_one(
    context: SimulationContext,
    run_config: RunConfig,
    model_kind: str,
    movement_scale: float,
) -> FitResult:
    model = _model(context, model_kind, movement_scale)
    fit = model.fit_em(
        context.observations,
        key=jax.random.PRNGKey(context.config.posterior_seed),
        n_iter=run_config.em_iterations,
        log_posterior_init=context.log_posterior_init,
        n_time_per_chunk=context.config.n_time,
        m_step_step_size=run_config.m_step_step_size,
        m_step_maxiter=run_config.m_step_maxiter,
        m_step_tol=run_config.m_step_tol,
        save_every=None,
        verboase=False,
    )
    posterior = _posterior_from_fit(fit, model_kind)
    tuning = np.asarray(fit["tuning"])
    fit_to_true = _best_posterior_overlap_permutation(
        posterior, context.state
    )
    aligned_posterior, _ = _apply_permutation(
        posterior, tuning, fit_to_true
    )
    (
        posthoc_posterior,
        posthoc_tuning,
        posthoc_dynamics,
        posthoc_log_marginal,
        posthoc_m_step_n_iter,
    ) = _posthoc_m_step_and_e_step(
        model,
        model_kind,
        context.observations,
        aligned_posterior,
        context.config.n_time,
    )
    label = (
        "jump"
        if model_kind == "jump"
        else f"no jump, movement scale {movement_scale:g}"
    )
    return FitResult(
        label=label,
        model_kind=model_kind,
        movement_scale=movement_scale,
        posterior=posthoc_posterior,
        tuning=posthoc_tuning,
        dynamics_posterior=posthoc_dynamics,
        fit_to_true=fit_to_true,
        legacy_log_marginal=float(fit["log_marginal"]),
        posthoc_log_marginal=posthoc_log_marginal,
        posthoc_m_step_n_iter=posthoc_m_step_n_iter,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(
        right_centered
    )
    if denominator <= 1e-12:
        return 0.0
    return float(left_centered @ right_centered / denominator)


def _latent_metrics(
    posterior: np.ndarray,
    state: np.ndarray,
    mask: np.ndarray,
) -> dict:
    error = np.abs(posterior.argmax(axis=1)[mask] - state[mask, 1])
    return {
        "mae_bins": float(error.mean()),
        "mae_normalized": float(error.mean() / posterior.shape[1]),
        "fraction_within_2_bins": float(np.mean(error <= 2)),
    }


def _tuning_metrics(
    tuning: np.ndarray,
    tuning_true: np.ndarray,
) -> dict:
    curve_correlations = np.asarray(
        [
            _safe_correlation(tuning[:, neuron], tuning_true[:, neuron])
            for neuron in range(tuning.shape[1])
        ]
    )
    population_correlations = np.asarray(
        [
            _safe_correlation(tuning[index], tuning_true[index])
            for index in range(tuning.shape[0])
        ]
    )
    relative_error = np.linalg.norm(tuning - tuning_true) / max(
        np.linalg.norm(tuning_true), 1e-12
    )
    return {
        "per_neuron_curve_correlation_median": float(
            np.median(curve_correlations)
        ),
        "per_neuron_curve_correlation_q10": float(
            np.quantile(curve_correlations, 0.1)
        ),
        "population_pattern_correlation_mean": float(
            population_correlations.mean()
        ),
        "relative_error": float(relative_error),
    }


def result_metrics(
    result: FitResult,
    context: SimulationContext,
) -> dict:
    continuous = context.state[:, 0] == 0
    fragmented = ~continuous
    metrics = {
        "continuous": _latent_metrics(
            result.posterior, context.state, continuous
        ),
        "fragmented": _latent_metrics(
            result.posterior, context.state, fragmented
        ),
        "tuning": _tuning_metrics(result.tuning, context.tuning_true),
        "legacy_log_marginal": result.legacy_log_marginal,
        "posthoc_log_marginal": result.posthoc_log_marginal,
        "posthoc_m_step_n_iter": result.posthoc_m_step_n_iter,
    }
    metrics["worst_state_mae_normalized"] = max(
        metrics["continuous"]["mae_normalized"],
        metrics["fragmented"]["mae_normalized"],
    )
    if result.dynamics_posterior is not None:
        metrics["fragmentation_accuracy"] = float(
            np.mean(
                result.dynamics_posterior.argmax(axis=1)
                == context.state[:, 0]
            )
        )
    return metrics


def longest_state_window(
    state: np.ndarray,
    dynamics_value: int,
    width: int = 180,
) -> tuple[int, int]:
    dynamics = state[:, 0].astype(int)
    boundaries = np.flatnonzero(dynamics[1:] != dynamics[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(dynamics)]
    candidates = [
        (stop - start, start, stop)
        for start, stop in zip(starts, stops)
        if dynamics[start] == dynamics_value
    ]
    _, segment_start, segment_stop = max(candidates)
    window_width = min(width, segment_stop - segment_start)
    start = segment_start + (segment_stop - segment_start - window_width) // 2
    return int(start), int(start + window_width)


def plot_latent_closeups(
    context: SimulationContext,
    results: list[FitResult],
    output: Path,
) -> None:
    windows = [
        ("longest continuous block", longest_state_window(context.state, 0)),
        ("longest fragmented block", longest_state_window(context.state, 1)),
    ]
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(12, 2.25 * len(results)),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    for row, result in enumerate(results):
        decoded = result.posterior.argmax(axis=1)
        for column, (window_name, (start, stop)) in enumerate(windows):
            ax = axes[row, column]
            time = np.arange(stop - start)
            truth = context.state[start:stop, 1]
            prediction = decoded[start:stop]
            if column == 0:
                ax.plot(time, truth, color="black", linewidth=2.0)
            else:
                ax.scatter(
                    time,
                    truth,
                    color="black",
                    s=11,
                    alpha=0.65,
                )
            ax.plot(
                time,
                prediction,
                color=colors[row % len(colors)],
                linewidth=1.0,
                alpha=0.9,
            )
            if row == 0:
                ax.set_title(
                    f"{window_name}\nabsolute time {start}-{stop}"
                )
            if column == 0:
                ax.set_ylabel(f"{result.label}\nlatent bin")
            ax.set_ylim(-3, context.config.n_latent_bin + 2)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time within block")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def largest_reset_windows(
    state: np.ndarray,
    n_windows: int = 3,
    before: int = 10,
    after: int = 25,
) -> list[tuple[int, int, int, int]]:
    candidates = np.flatnonzero(state[:, 0] == 1)
    candidates = candidates[
        (candidates >= before) & (candidates + after < len(state))
    ]
    candidates = np.asarray(
        [
            index
            for index in candidates
            if np.sum(state[index - before : index + after, 0]) == 1
        ],
        dtype=int,
    )
    jump_size = np.abs(
        state[candidates, 1] - state[candidates - 1, 1]
    )
    selected = []
    for index in candidates[np.argsort(jump_size)[::-1]]:
        if all(abs(int(index) - event) >= before + after for event in selected):
            selected.append(int(index))
        if len(selected) == n_windows:
            break
    return [
        (
            event - before,
            event + after,
            event,
            int(abs(state[event, 1] - state[event - 1, 1])),
        )
        for event in sorted(selected)
    ]


def plot_reset_event_closeups(
    context: SimulationContext,
    results: list[FitResult],
    output: Path,
) -> None:
    windows = largest_reset_windows(context.state)
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(
        len(results),
        len(windows),
        figsize=(14, 2.15 * len(results)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, result in enumerate(results):
        decoded = result.posterior.argmax(axis=1)
        for column, (start, stop, event, jump_size) in enumerate(windows):
            ax = axes[row, column]
            time = np.arange(start, stop) - event
            ax.plot(
                time,
                context.state[start:stop, 1],
                color="black",
                linewidth=2.0,
                label="true latent",
            )
            ax.plot(
                time,
                decoded[start:stop],
                color=colors[row % len(colors)],
                linewidth=1.2,
                alpha=0.9,
                label="inferred MAP",
            )
            fragmented = context.state[start:stop, 0] == 1
            for fragmented_time in time[fragmented]:
                ax.axvspan(
                    fragmented_time - 0.5,
                    fragmented_time + 0.5,
                    color="0.8",
                    alpha=0.7,
                    linewidth=0,
                )
            if row == 0:
                ax.set_title(
                    f"reset at t={event}; true step={jump_size} bins"
                )
            if column == 0:
                short_label = (
                    "jump"
                    if result.model_kind == "jump"
                    else f"no jump\nscale {result.movement_scale:g}"
                )
                ax.set_ylabel(f"{short_label}\nlatent bin")
            ax.set_ylim(-3, context.config.n_latent_bin + 2)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time relative to reset")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_population(
    context: SimulationContext,
    jump_result: FitResult,
    best_nojump: FitResult,
    output: Path,
) -> None:
    neuron_order = np.argsort(np.argmax(context.tuning_true, axis=0))
    conditions = [
        ("true tuning", context.tuning_true),
        ("jump", jump_result.tuning),
        (best_nojump.label, best_nojump.tuning),
    ]
    displays = [
        np.log1p(tuning[:, neuron_order].T)
        for _, tuning in conditions
    ]
    vmax = float(
        np.quantile(np.concatenate([values.ravel() for values in displays]), 0.995)
    )
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
    for ax, (label, _), values in zip(axes, conditions, displays):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.set_ylabel("neurons")
    axes[-1].set_xlabel("latent bin")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("log(1 + firing rate)")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_examples(
    context: SimulationContext,
    jump_result: FitResult,
    best_nojump: FitResult,
    output: Path,
) -> None:
    jump_correlations = np.asarray(
        [
            _safe_correlation(
                jump_result.tuning[:, neuron],
                context.tuning_true[:, neuron],
            )
            for neuron in range(context.config.n_neuron)
        ]
    )
    nojump_correlations = np.asarray(
        [
            _safe_correlation(
                best_nojump.tuning[:, neuron],
                context.tuning_true[:, neuron],
            )
            for neuron in range(context.config.n_neuron)
        ]
    )
    ordered = np.argsort(nojump_correlations)
    selected = ordered[
        np.rint(np.linspace(0.1, 0.9, 6) * (len(ordered) - 1)).astype(int)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, neuron in zip(axes.flat, selected):
        ax.plot(
            context.tuning_true[:, neuron],
            color="black",
            linewidth=1.5,
            label="true",
        )
        ax.plot(
            jump_result.tuning[:, neuron],
            color="C0",
            linewidth=1.2,
            label="jump",
        )
        ax.plot(
            best_nojump.tuning[:, neuron],
            color="C3",
            linewidth=1.2,
            linestyle="--",
            label=f"best no-jump (scale {best_nojump.movement_scale:g})",
        )
        ax.set_title(
            f"neuron {neuron}: "
            f"r={jump_correlations[neuron]:.2f} jump, "
            f"{nojump_correlations[neuron]:.2f} no-jump"
        )
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("latent bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("firing rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_raw_results(
    context: SimulationContext,
    results: list[FitResult],
    output: Path,
) -> None:
    arrays = {
        "state": context.state,
        "observations": context.observations,
        "tuning_true": context.tuning_true,
        "log_posterior_init": context.log_posterior_init,
    }
    for result in results:
        prefix = result.label.replace(" ", "_").replace(",", "")
        arrays[f"{prefix}__posterior"] = result.posterior
        arrays[f"{prefix}__tuning"] = result.tuning
        arrays[f"{prefix}__fit_to_true"] = result.fit_to_true
        if result.dynamics_posterior is not None:
            arrays[f"{prefix}__dynamics_posterior"] = (
                result.dynamics_posterior
            )
    np.savez_compressed(output, **arrays)


def run_experiment(
    simulation_config: SimulationConfig,
    run_config: RunConfig,
    output_dir: Path,
    environment: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = build_simulation(simulation_config)
    results = []
    print("fit=jump", flush=True)
    results.append(
        fit_one(
            context,
            run_config,
            "jump",
            simulation_config.movement_variance,
        )
    )
    for movement_scale in run_config.nojump_movement_scales:
        print(f"fit=nojump movement_scale={movement_scale:g}", flush=True)
        results.append(
            fit_one(
                context,
                run_config,
                "nojump",
                movement_scale,
            )
        )
    metrics = {
        result.label: result_metrics(result, context)
        for result in results
    }
    jump_result = results[0]
    best_nojump = min(
        results[1:],
        key=lambda result: (
            metrics[result.label]["worst_state_mae_normalized"],
            -metrics[result.label]["tuning"][
                "per_neuron_curve_correlation_median"
            ],
        ),
    )
    jump_metrics = metrics[jump_result.label]
    best_metrics = metrics[best_nojump.label]
    summary = {
        "simulation": asdict(simulation_config),
        "run": asdict(run_config),
        "environment": environment,
        "same_initial_posterior_for_all_models": True,
        "oracle_pipeline": (
            "all-time posterior-overlap permutation, one fixed-posterior Adam "
            "M-step, one diagnostic E-step"
        ),
        "actual_fragmented_fraction": float(
            np.mean(context.state[:, 0] == 1)
        ),
        "mean_observed_spikes_per_neuron_per_bin": float(
            context.observations.mean()
        ),
        "metrics": metrics,
        "best_nojump_by_minimax_state_error": best_nojump.label,
        "separation": {
            "jump_worst_state_mae_normalized": jump_metrics[
                "worst_state_mae_normalized"
            ],
            "best_nojump_worst_state_mae_normalized": best_metrics[
                "worst_state_mae_normalized"
            ],
            "worst_state_mae_ratio": float(
                best_metrics["worst_state_mae_normalized"]
                / max(jump_metrics["worst_state_mae_normalized"], 1e-12)
            ),
            "jump_tuning_median_correlation": jump_metrics["tuning"][
                "per_neuron_curve_correlation_median"
            ],
            "best_nojump_tuning_median_correlation": best_metrics["tuning"][
                "per_neuron_curve_correlation_median"
            ],
        },
    }
    plot_latent_closeups(
        context, results, output_dir / "latent_state_closeups.png"
    )
    plot_reset_event_closeups(
        context, results, output_dir / "reset_event_closeups.png"
    )
    plot_tuning_population(
        context,
        jump_result,
        best_nojump,
        output_dir / "tuning_population.png",
    )
    plot_tuning_examples(
        context,
        jump_result,
        best_nojump,
        output_dir / "tuning_examples.png",
    )
    save_raw_results(context, results, output_dir / "results.npz")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for result in results:
        if not np.array_equal(
            np.sort(result.fit_to_true),
            np.arange(simulation_config.n_latent_bin),
        ):
            raise AssertionError(f"non-bijective permutation for {result.label}")
    for filename in (
        "latent_state_closeups.png",
        "reset_event_closeups.png",
        "tuning_population.png",
        "tuning_examples.png",
        "results.npz",
        "summary.json",
    ):
        if (output_dir / filename).stat().st_size < 1_000:
            raise AssertionError(f"missing or empty artifact: {filename}")
    print(f"saved_summary={summary_path}", flush=True)
    print(
        "separation="
        + json.dumps(summary["separation"], sort_keys=True),
        flush=True,
    )
    return summary


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if not scales:
        raise argparse.ArgumentTypeError("provide comma-separated scales")
    return scales


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--n-neuron", type=int, default=60)
    parser.add_argument("--n-time", type=int, default=8_000)
    parser.add_argument("--n-latent-bin", type=int, default=100)
    parser.add_argument("--movement-variance", type=float, default=0.35)
    parser.add_argument("--tuning-lengthscale", type=float, default=6.0)
    parser.add_argument("--p-move-to-jump", type=float, default=0.06)
    parser.add_argument("--p-jump-to-move", type=float, default=0.02)
    parser.add_argument("--rate-bias", type=float, default=-0.5)
    parser.add_argument("--simulation-seed", type=int, default=103)
    parser.add_argument("--generative-param-seed", type=int, default=123)
    parser.add_argument("--fitted-param-seed", type=int, default=124)
    parser.add_argument("--posterior-seed", type=int, default=5)
    parser.add_argument("--em-iterations", type=int, default=20)
    parser.add_argument("--m-step-step-size", type=float, default=0.01)
    parser.add_argument("--m-step-maxiter", type=int, default=1_000)
    parser.add_argument("--m-step-tol", type=float, default=1e-10)
    parser.add_argument(
        "--nojump-movement-scales",
        type=_parse_scales,
        default=_parse_scales("0.35,3,10,30"),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    environment = preflight(require_gpu=not args.allow_cpu)
    simulation_config = SimulationConfig(
        name=args.name,
        n_neuron=args.n_neuron,
        n_time=args.n_time,
        n_latent_bin=args.n_latent_bin,
        movement_variance=args.movement_variance,
        tuning_lengthscale=args.tuning_lengthscale,
        p_move_to_jump=args.p_move_to_jump,
        p_jump_to_move=args.p_jump_to_move,
        rate_bias=args.rate_bias,
        simulation_seed=args.simulation_seed,
        generative_param_seed=args.generative_param_seed,
        fitted_param_seed=args.fitted_param_seed,
        posterior_seed=args.posterior_seed,
    )
    run_config = RunConfig(
        em_iterations=args.em_iterations,
        m_step_step_size=args.m_step_step_size,
        m_step_maxiter=args.m_step_maxiter,
        m_step_tol=args.m_step_tol,
        nojump_movement_scales=args.nojump_movement_scales,
    )
    return run_experiment(
        simulation_config,
        run_config,
        args.output_dir,
        environment,
    )


if __name__ == "__main__":
    main()
