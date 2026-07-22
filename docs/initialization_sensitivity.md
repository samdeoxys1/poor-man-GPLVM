# Parameter-initialization sensitivity diagnostic

`poor_man_gplvm.diagnostics.initialization_sensitivity` tests why changing the
initial tuning parameters can change a Poisson GPLVM EM fit even when the latent
posterior initialization is held fixed.

The diagnostic uses Adam only. It first freezes both the simulated observations
and posterior, then runs the current finite/stopped M-step from several parameter
seeds. It repeats the same fixed-posterior comparison with a forced-long Adam run
(`tol=-1`, so it reaches the requested iteration limit), and finally follows the
pairwise tuning and posterior differences through the normal EM loop. This
distinguishes residual M-step error from amplification by later E/M alternation.

Defaults match `poor_man_gplvm/notebooks/for_figures/[Figure] simulation.ipynb`:
100 neurons, 10,000 time bins, 100 latent bins, tuning lengthscale 10, movement
variance 1, and simulation seed 103. Seed 123 is included deliberately: the
notebook uses the class default seed 123 for the generative model, so fitting with
parameter seed 123 starts from the exact generative tuning parameters.

Run on a configured GPU environment:

```bash
python -m poor_man_gplvm.diagnostics.initialization_sensitivity \
  --output-dir /path/to/output
```

The output directory contains `summary.json`, compressed raw arrays, and plots
for fixed-posterior convergence, current versus forced-long Adam, and EM drift.
The script requires a JAX GPU and `ptxas` by default; `--allow-cpu` is available
only for explicitly requested CPU diagnostics.
