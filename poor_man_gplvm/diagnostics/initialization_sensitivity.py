"""Diagnose parameter-initialization sensitivity in Poisson GPLVM EM fits.

The diagnostic deliberately uses the package's Adam M-step throughout.  It
separates two questions that are otherwise entangled in a complete EM fit:

1. With observations and posterior held fixed, do different parameter starts
   reach the same M-step solution?
2. If the first M-step leaves a small difference, does alternating with the
   E-step amplify that difference over subsequent EM iterations?

The default simulation matches ``[Figure] simulation.ipynb``: 100 neurons,
10,000 time bins, 100 latent bins, tuning lengthscale 10, movement variance 1,
and simulation seed 103.  Parameter seed 123 is intentionally included because
it is also the generative model's default seed in that notebook.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import platform
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import matplotlib.pyplot as plt
import numpy as np

from poor_man_gplvm import PoissonGPLVMJump1D
import poor_man_gplvm.fit_tuning_helper as fth


@dataclass(frozen=True)
class ExperimentConfig:
    n_neuron: int = 100
    n_time: int = 10_000
    n_latent_bin: int = 100
    movement_variance: float = 1.0
    tuning_lengthscale: float = 10.0
    simulation_seed: int = 103
    generative_param_seed: int = 123
    posterior_seed: int = 5
    posterior_random_scale: float = 1.0
    param_init_seeds: tuple[int, ...] = (123, 0, 1, 2)
    em_iterations: int = 20
    m_step_step_size: float = 0.01
    m_step_maxiter: int = 1_000
    m_step_tol: float = 1e-10
    forced_long_maxiter: int = 5_000
    n_time_per_chunk: int = 10_000


@dataclass
class SimulationContext:
    config: ExperimentConfig
    state: np.ndarray
    observations: jax.Array
    tuning_true: np.ndarray
    log_posterior_init: jax.Array
    model_kwargs: dict


def _model_for_seed(context: SimulationContext, seed: int) -> PoissonGPLVMJump1D:
    return PoissonGPLVMJump1D(rng_init_int=seed, **context.model_kwargs)


def build_simulation(config: ExperimentConfig) -> SimulationContext:
    """Build the notebook-matched simulation and one shared posterior start."""
    model_kwargs = {
        "n_neuron": config.n_neuron,
        "n_latent_bin": config.n_latent_bin,
        "movement_variance": config.movement_variance,
        "tuning_lengthscale": config.tuning_lengthscale,
    }
    generative_model = PoissonGPLVMJump1D(
        rng_init_int=config.generative_param_seed, **model_kwargs
    )
    state, observations = generative_model.sample(
        config.n_time, key=jax.random.PRNGKey(config.simulation_seed)
    )
    posterior_model = PoissonGPLVMJump1D(rng_init_int=0, **model_kwargs)
    log_posterior_init, _ = posterior_model.init_latent_posterior(
        config.n_time,
        jax.random.PRNGKey(config.posterior_seed),
        random_scale=config.posterior_random_scale,
    )
    return SimulationContext(
        config=config,
        state=np.asarray(state),
        observations=observations,
        tuning_true=np.asarray(generative_model.tuning),
        log_posterior_init=log_posterior_init,
        model_kwargs=model_kwargs,
    )
def _objective_and_gradient(
    params: jax.Array,
    hyperparam: dict,
    basis: jax.Array,
    y_weighted: jax.Array,
    t_weighted: jax.Array,
) -> tuple[float, float]:
    loss, gradient = jax.value_and_grad(fth.poisson_m_step_objective)(
        params, hyperparam, basis, y_weighted, t_weighted
    )
    return float(loss), float(fth.tree_l2_norm(gradient))


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=float)))))


def _relative_rmse(left: np.ndarray, right: np.ndarray) -> float:
    scale = 0.5 * (_rms(left) + _rms(right))
    return _rms(np.asarray(left) - np.asarray(right)) / max(scale, 1e-12)


def _pairwise_records(
    arrays: Mapping[str, np.ndarray], metric
) -> list[dict]:
    records = []
    for left, right in itertools.combinations(arrays, 2):
        records.append(
            {
                "left": left,
                "right": right,
                "value": float(metric(arrays[left], arrays[right])),
            }
        )
    return records


def _aggregate_pairwise(records: Sequence[dict]) -> dict:
    values = np.asarray([record["value"] for record in records], dtype=float)
    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "pairs": list(records),
    }


def _posterior_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(left - right).sum(axis=1)))


def _map_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left.argmax(axis=1) != right.argmax(axis=1)))


def _make_runner(step_size: float, maxiter: int, tol: float):
    return fth.make_adam_runner(
        fth.poisson_m_step_objective,
        step_size=step_size,
        maxiter=maxiter,
        tol=tol,
    )


def _decode_latent_marginal(
    model: PoissonGPLVMJump1D,
    observations: jax.Array,
    tuning: jax.Array,
    n_time_per_chunk: int,
) -> tuple[np.ndarray, jax.Array]:
    decode_result = model.decode_latent(
        observations,
        tuning=tuning,
        n_time_per_chunk=n_time_per_chunk,
    )
    posterior = np.asarray(decode_result["posterior_latent_marg"])
    log_posterior = jscipy.special.logsumexp(
        jnp.asarray(decode_result["log_posterior_all"]), axis=1
    )
    return posterior, log_posterior


def run_fixed_posterior_m_steps(
    context: SimulationContext,
    maxiter: int,
    tol: float,
) -> dict:
    """Run one M-step per parameter seed with exactly the same posterior."""
    config = context.config
    hyperparam = {"param_prior_std": 1.0, "smoothness_penalty": 0.0}
    first_model = _model_for_seed(context, config.param_init_seeds[0])
    basis = first_model.tuning_basis
    y_weighted, t_weighted = fth.get_statistics(
        context.log_posterior_init, context.observations
    )
    runner, init_optimizer = _make_runner(
        config.m_step_step_size, maxiter=maxiter, tol=tol
    )

    params = {}
    tunings = {}
    posteriors = {}
    per_seed = {}
    loss_histories = {}
    gradient_histories = {}

    for seed in config.param_init_seeds:
        label = str(seed)
        model = _model_for_seed(context, seed)
        params_init = model.params
        result = runner(
            params_init,
            init_optimizer(params_init),
            hyperparam,
            basis,
            y_weighted,
            t_weighted,
        )
        fitted_params = result["params"]
        tuning = model.get_tuning(fitted_params, hyperparam, basis)
        posterior, _ = _decode_latent_marginal(
            model,
            context.observations,
            tuning,
            config.n_time_per_chunk,
        )
        actual_loss, actual_gradient_norm = _objective_and_gradient(
            fitted_params, hyperparam, basis, y_weighted, t_weighted
        )
        n_iter = int(result["n_iter"])

        params[label] = np.asarray(fitted_params)
        tunings[label] = np.asarray(tuning)
        posteriors[label] = posterior
        loss_histories[label] = np.asarray(result["loss_history"][:n_iter])
        gradient_histories[label] = np.asarray(result["error_history"][:n_iter])
        per_seed[label] = {
            "n_iter_reported": n_iter,
            "hit_iteration_limit": bool(n_iter >= maxiter),
            "actual_final_loss": actual_loss,
            "actual_final_gradient_norm": actual_gradient_norm,
            "reported_final_loss": float(result["final_loss"]),
            "reported_final_gradient_norm": float(result["final_error"]),
            "param_rmse_from_start": _rms(np.asarray(fitted_params - params_init)),
            "tuning_relative_rmse_from_truth": _relative_rmse(
                np.asarray(tuning), context.tuning_true
            ),
        }

    metrics = {
        "params_relative_rmse": _aggregate_pairwise(
            _pairwise_records(params, _relative_rmse)
        ),
        "tuning_relative_rmse": _aggregate_pairwise(
            _pairwise_records(tunings, _relative_rmse)
        ),
        "posterior_mean_total_variation": _aggregate_pairwise(
            _pairwise_records(posteriors, _posterior_tv)
        ),
        "posterior_map_disagreement": _aggregate_pairwise(
            _pairwise_records(posteriors, _map_disagreement)
        ),
    }
    losses = np.asarray(
        [per_seed[str(seed)]["actual_final_loss"] for seed in config.param_init_seeds]
    )
    metrics["objective_span"] = float(losses.max() - losses.min())
    return {
        "per_seed": per_seed,
        "metrics": metrics,
        "params": params,
        "tunings": tunings,
        "posteriors": posteriors,
        "loss_histories": loss_histories,
        "gradient_histories": gradient_histories,
    }


def run_em_drift(context: SimulationContext) -> dict:
    """Mirror the production Adam EM loop while retaining only drift summaries."""
    config = context.config
    hyperparam = {"param_prior_std": 1.0, "smoothness_penalty": 0.0}
    runner, init_optimizer = _make_runner(
        config.m_step_step_size,
        maxiter=config.m_step_maxiter,
        tol=config.m_step_tol,
    )
    models = {
        str(seed): _model_for_seed(context, seed) for seed in config.param_init_seeds
    }
    params = {label: model.params for label, model in models.items()}
    optimizer_states = {
        label: init_optimizer(value) for label, value in params.items()
    }
    log_posteriors = {
        label: context.log_posterior_init for label in models
    }
    basis = next(iter(models.values())).tuning_basis

    trajectory = {
        "params_relative_rmse_mean": [],
        "params_relative_rmse_max": [],
        "tuning_relative_rmse_mean": [],
        "tuning_relative_rmse_max": [],
        "posterior_mean_total_variation_mean": [],
        "posterior_mean_total_variation_max": [],
        "posterior_map_disagreement_mean": [],
        "posterior_map_disagreement_max": [],
        "m_step_iterations": [],
        "m_step_actual_gradient_norm": [],
        "m_step_actual_loss": [],
    }

    for _ in range(config.em_iterations):
        tunings = {}
        posterior_marginals = {}
        iteration_counts = []
        gradient_norms = []
        losses = []

        for label, model in models.items():
            y_weighted, t_weighted = fth.get_statistics(
                log_posteriors[label], context.observations
            )
            result = runner(
                params[label],
                optimizer_states[label],
                hyperparam,
                basis,
                y_weighted,
                t_weighted,
            )
            params[label] = result["params"]
            optimizer_states[label] = result["opt_state"]
            tuning = model.get_tuning(params[label], hyperparam, basis)
            posterior, log_posterior = _decode_latent_marginal(
                model,
                context.observations,
                tuning,
                config.n_time_per_chunk,
            )
            log_posteriors[label] = log_posterior
            loss, gradient_norm = _objective_and_gradient(
                params[label], hyperparam, basis, y_weighted, t_weighted
            )
            tunings[label] = np.asarray(tuning)
            posterior_marginals[label] = posterior
            iteration_counts.append(int(result["n_iter"]))
            gradient_norms.append(gradient_norm)
            losses.append(loss)

        pairwise_groups = {
            "params_relative_rmse": _pairwise_records(params, _relative_rmse),
            "tuning_relative_rmse": _pairwise_records(tunings, _relative_rmse),
            "posterior_mean_total_variation": _pairwise_records(
                posterior_marginals, _posterior_tv
            ),
            "posterior_map_disagreement": _pairwise_records(
                posterior_marginals, _map_disagreement
            ),
        }
        for name, records in pairwise_groups.items():
            aggregate = _aggregate_pairwise(records)
            trajectory[f"{name}_mean"].append(aggregate["mean"])
            trajectory[f"{name}_max"].append(aggregate["max"])
        trajectory["m_step_iterations"].append(iteration_counts)
        trajectory["m_step_actual_gradient_norm"].append(gradient_norms)
        trajectory["m_step_actual_loss"].append(losses)

    return {key: np.asarray(value) for key, value in trajectory.items()}


def _fixed_summary(result: dict) -> dict:
    return {"per_seed": result["per_seed"], "metrics": result["metrics"]}


def build_summary(
    context: SimulationContext,
    current: dict,
    forced_long: dict,
    em_trajectory: dict,
) -> dict:
    current_tuning = current["metrics"]["tuning_relative_rmse"]["mean"]
    forced_tuning = forced_long["metrics"]["tuning_relative_rmse"]["mean"]
    first_em_tv = float(em_trajectory["posterior_mean_total_variation_mean"][0])
    final_em_tv = float(em_trajectory["posterior_mean_total_variation_mean"][-1])
    current_gradients = np.asarray(
        [v["actual_final_gradient_norm"] for v in current["per_seed"].values()]
    )
    forced_gradients = np.asarray(
        [v["actual_final_gradient_norm"] for v in forced_long["per_seed"].values()]
    )
    return {
        "config": asdict(context.config),
        "environment": {
            "hostname": platform.node(),
            "python": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "ptxas": shutil.which("ptxas"),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "important_setup_fact": (
            "Parameter seed 123 exactly matches the generative model seed; it is an "
            "oracle parameter initialization in this simulation."
        ),
        "fixed_posterior_current_adam": _fixed_summary(current),
        "fixed_posterior_forced_long_adam": _fixed_summary(forced_long),
        "diagnostic_ratios": {
            "forced_over_current_pairwise_tuning_difference": float(
                forced_tuning / max(current_tuning, 1e-30)
            ),
            "forced_over_current_mean_gradient_norm": float(
                forced_gradients.mean() / max(current_gradients.mean(), 1e-30)
            ),
            "final_over_first_em_posterior_tv": float(
                final_em_tv / max(first_em_tv, 1e-30)
            ),
        },
        "em_drift": {
            key: np.asarray(value).tolist() for key, value in em_trajectory.items()
        },
    }


def plot_results(
    current: dict,
    forced_long: dict,
    em_trajectory: dict,
    output_dir: Path,
) -> list[str]:
    """Plot convergence and amplification summaries from computed results."""
    output_paths = []
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for index, (label, history) in enumerate(
        forced_long["loss_histories"].items()
    ):
        history = np.asarray(history)
        axes[0].plot(
            history - np.nanmin(history),
            label=f"seed {label}",
            color=colors[index % len(colors)],
        )
        axes[1].semilogy(
            forced_long["gradient_histories"][label],
            color=colors[index % len(colors)],
        )
    axes[0].set(xlabel="Adam update", ylabel="loss - within-run minimum")
    axes[1].set(xlabel="Adam update", ylabel="reported gradient norm")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "fixed_posterior_forced_long_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(str(path))

    names = [
        "tuning_relative_rmse",
        "posterior_mean_total_variation",
        "posterior_map_disagreement",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(
        x - width / 2,
        [current["metrics"][name]["mean"] for name in names],
        width,
        label="current Adam",
    )
    ax.bar(
        x + width / 2,
        [forced_long["metrics"][name]["mean"] for name in names],
        width,
        label="forced-long Adam",
    )
    ax.set_xticks(x, ["tuning rel. RMSE", "posterior TV", "MAP disagreement"])
    ax.set_ylabel("mean across parameter-seed pairs")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "fixed_posterior_current_vs_forced_long.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(str(path))

    iterations = np.arange(1, len(em_trajectory["tuning_relative_rmse_mean"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(
        iterations,
        em_trajectory["tuning_relative_rmse_mean"],
        marker="o",
        label="tuning rel. RMSE",
    )
    axes[0].plot(
        iterations,
        em_trajectory["posterior_mean_total_variation_mean"],
        marker="o",
        label="posterior TV",
    )
    axes[0].set(xlabel="EM iteration", ylabel="mean pairwise difference")
    axes[0].legend(frameon=False)
    for column in range(em_trajectory["m_step_actual_gradient_norm"].shape[1]):
        axes[1].semilogy(
            iterations,
            em_trajectory["m_step_actual_gradient_norm"][:, column],
            marker="o",
        )
    axes[1].set(xlabel="EM iteration", ylabel="actual final gradient norm")
    fig.tight_layout()
    path = output_dir / "em_drift_and_m_step_gradient.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(str(path))
    return output_paths


def save_raw_results(
    output_dir: Path,
    current: dict,
    forced_long: dict,
    em_trajectory: dict,
) -> str:
    arrays = {f"em__{key}": value for key, value in em_trajectory.items()}
    for prefix, result in (("current", current), ("forced_long", forced_long)):
        for seed, value in result["params"].items():
            arrays[f"{prefix}__params__seed_{seed}"] = value
        for seed, value in result["tunings"].items():
            arrays[f"{prefix}__tuning__seed_{seed}"] = value
        for seed, value in result["loss_histories"].items():
            arrays[f"{prefix}__loss_history__seed_{seed}"] = value
        for seed, value in result["gradient_histories"].items():
            arrays[f"{prefix}__gradient_history__seed_{seed}"] = value
    path = output_dir / "raw_results.npz"
    np.savez_compressed(path, **arrays)
    return str(path)


def preflight(require_gpu: bool = True) -> dict:
    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, but JAX devices are {devices}")
    if require_gpu and shutil.which("ptxas") is None:
        raise RuntimeError("GPU required, but ptxas is not available")
    return {"devices": [str(device) for device in devices]}


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = build_simulation(config)
    print("stage=fixed_posterior_current_adam", flush=True)
    current = run_fixed_posterior_m_steps(
        context, maxiter=config.m_step_maxiter, tol=config.m_step_tol
    )
    print("stage=fixed_posterior_forced_long_adam", flush=True)
    forced_long = run_fixed_posterior_m_steps(
        context, maxiter=config.forced_long_maxiter, tol=-1.0
    )
    print("stage=em_drift", flush=True)
    em_trajectory = run_em_drift(context)
    summary = build_summary(context, current, forced_long, em_trajectory)
    summary["artifacts"] = {
        "figures": plot_results(current, forced_long, em_trajectory, output_dir),
        "raw_results": save_raw_results(
            output_dir, current, forced_long, em_trajectory
        ),
    }
    summary_path = output_dir / "summary.json"
    summary["artifacts"]["summary"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"saved_summary={summary_path}", flush=True)
    return summary


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("provide at least two comma-separated seeds")
    return seeds


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@contextlib.contextmanager
def _output_heartbeat(enabled: bool, interval_seconds: float = 5.0):
    """Keep ``jupyter run`` alive while a compiled stage produces no output."""
    if not enabled:
        yield
        return
    stop = threading.Event()

    def emit() -> None:
        while not stop.wait(interval_seconds):
            print(f"heartbeat_monotonic_seconds={time.monotonic():.1f}", flush=True)

    thread = threading.Thread(target=emit, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    output_dir_default = os.environ.get("PMG_INIT_SENSITIVITY_OUTPUT_DIR")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(output_dir_default) if output_dir_default else None,
        required=output_dir_default is None,
    )
    parser.add_argument(
        "--param-init-seeds",
        type=_parse_seeds,
        default=_parse_seeds(
            os.environ.get("PMG_INIT_SENSITIVITY_PARAM_SEEDS", "123,0,1,2")
        ),
    )
    parser.add_argument(
        "--n-neuron", type=int, default=_env_int("PMG_INIT_SENSITIVITY_N_NEURON", 100)
    )
    parser.add_argument(
        "--n-time", type=int, default=_env_int("PMG_INIT_SENSITIVITY_N_TIME", 10_000)
    )
    parser.add_argument(
        "--n-latent-bin",
        type=int,
        default=_env_int("PMG_INIT_SENSITIVITY_N_LATENT_BIN", 100),
    )
    parser.add_argument(
        "--em-iterations",
        type=int,
        default=_env_int("PMG_INIT_SENSITIVITY_EM_ITERATIONS", 20),
    )
    parser.add_argument(
        "--m-step-maxiter",
        type=int,
        default=_env_int("PMG_INIT_SENSITIVITY_M_STEP_MAXITER", 1_000),
    )
    parser.add_argument(
        "--m-step-tol",
        type=float,
        default=_env_float("PMG_INIT_SENSITIVITY_M_STEP_TOL", 1e-10),
    )
    parser.add_argument(
        "--forced-long-maxiter",
        type=int,
        default=_env_int("PMG_INIT_SENSITIVITY_FORCED_LONG_MAXITER", 5_000),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow a CPU diagnostic; GPU is required by default",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    preflight(require_gpu=not args.allow_cpu)
    config = ExperimentConfig(
        n_neuron=args.n_neuron,
        n_time=args.n_time,
        n_latent_bin=args.n_latent_bin,
        param_init_seeds=args.param_init_seeds,
        em_iterations=args.em_iterations,
        m_step_maxiter=args.m_step_maxiter,
        m_step_tol=args.m_step_tol,
        forced_long_maxiter=args.forced_long_maxiter,
        n_time_per_chunk=min(args.n_time, 10_000),
    )
    return run_experiment(config, args.output_dir)


if __name__ == "__main__":
    # ``jupyter run`` executes the file inside ipykernel, whose own ``-f`` and
    # connection-file arguments remain in sys.argv. Configuration for that
    # launch mode comes from the PMG_INIT_SENSITIVITY_* environment variables.
    launched_by_ipykernel = "ipykernel_launcher" in Path(sys.argv[0]).name
    argv = [] if launched_by_ipykernel else None
    with _output_heartbeat(enabled=launched_by_ipykernel):
        main(argv)
