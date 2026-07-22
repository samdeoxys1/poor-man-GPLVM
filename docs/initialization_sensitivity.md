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

The CLI defaults also have matching `PMG_INIT_SENSITIVITY_*` environment
variables; their names are visible in `build_arg_parser()`.

## Findings in the notebook-matched simulation

The following values came from the default full diagnostic with parameter seeds
`123, 0, 1, 2`.

- The current M-steps stopped after 613--930 updates, before the 1,000-update
  cap, with actual gradient norms of 19--163. At a loss around 833,500, float32
  spacing is 0.0625, while `tol * loss` for `tol=1e-10` is about 0.000083. The
  stopping rule therefore cannot observe a nonzero change that small: it stops
  when one loss update happens to round to exact equality.
- On the identical fixed posterior, current Adam left mean pairwise tuning
  relative RMSE 0.00500. That small tuning difference already produced mean
  posterior total variation 0.194 after the first E-step.
- Forcing exactly 1,000 Adam updates reduced the same values to 0.00267 and
  0.101. Forcing 5,000 updates reduced them to 0.000592 and 0.0110. Thus both
  the premature equality stop and the total optimization budget matter.
- The normal EM loop amplified mean posterior total variation from 0.194 after
  iteration 1 to 0.932 after iteration 20. Even with 5,000 forced Adam updates
  at every M-step, it grew from 0.0110 to 0.365 by iteration 5; the residual
  M-step error is small but the alternating fit is highly sensitive to it.
- Parameter differences are much larger than tuning differences partly because
  the 18-column tuning basis is ill-conditioned (condition number about 1,306).
  The explicit bias column has absolute cosine 0.932 with the leading RBF basis
  column. Coefficients can therefore move substantially in a nearly redundant
  direction while changing the represented tuning curve very little.
- Seed 123 is special in this simulation: it is also the generative model's
  parameter seed, so it starts from the true tuning parameters. This makes the
  notebook comparison an oracle initialization versus unrelated random starts,
  not four exchangeable parameter starts.
- Separately, `AbstractGPLVMJump1D.init_latent_posterior` multiplies positive
  uniform random draws by `random_scale` and then row-normalizes them. Any
  positive scale therefore cancels exactly: for a fixed key, `random_scale=1`
  and `random_scale=0.1` give the same posterior initialization. The key, not
  this scale argument, controls the current jump-model posterior start.

The M-step objective is convex in the default RBF/Poisson setup, and the Gaussian
parameter prior makes it strictly convex. Exact M-step optimization would erase
parameter initialization after the first fixed-posterior M-step. The observed
dependence is numerical: the implemented Adam M-step stops far from that unique
solution, poor basis conditioning slows coefficient convergence, and the E-step
then amplifies the remaining tuning differences into distinct EM trajectories.
