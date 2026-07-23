# Jump versus no-jump simulation diagnostic

`poor_man_gplvm.diagnostics.jump_model_separation` tests whether an explicit
jump state remains necessary after granting every fitted model the same strong,
simulation-only rescue:

1. the same random posterior initialization;
2. an all-time oracle posterior-overlap permutation;
3. one fixed-posterior Adam M-step after permutation; and
4. one diagnostic E-step with the refitted tuning.

No fit is initialized from the true latent or true parameters. The generative
parameter seed is `123`, the fitted parameter seed is `124`, and the shared
posterior seed is `5`. Adam is deliberate here: the public M-step avoids the
memory problems encountered with LBFGS.

## Preferred non-isolated regime

The default simulation uses 200 neurons, the first 3,500 bins of a fixed
8,000-bin source simulation, 100 latent bins, continuous movement scale `0.35`,
tuning lengthscale `5`, and a `-1.0` shift of the generative bias parameters.
Its dynamics probabilities are `p_move_to_jump=0.04` and
`p_jump_to_move=0.40`.

The realized fragmented fraction is `0.0889`. Fragmented bursts have mean
length `2.51`, median length `2`, and maximum length `13`, so this is not an
isolated-reset simulation. The mean count is `0.440` spikes per neuron per bin.

After 20 EM iterations and oracle alignment/refit:

| model | continuous latent MAE | fragmented latent MAE | median tuning correlation |
|---|---:|---:|---:|
| jump | 0.00145 | 0.00952 | 0.928 |
| no jump, scale 0.35 | 0.213 | 0.262 | 0.715 |
| no jump, scale 3 | 0.0360 | 0.203 | 0.857 |
| no jump, scale 10 | 0.00823 | 0.145 | 0.882 |
| no jump, scale 30 | 0.00795 | 0.0398 | 0.884 |

Scale `30` is the best tested no-jump model by minimax state-specific error,
but its fragmented error is still 4.19 times the jump model's error. The jump
fit is essentially exact in both states. Its tuning is not perfectly smooth,
but it preserves the true qualitative shape; the best no-jump tuning is more
jagged and contains local peak distortions.

`fragmented_burst_closeups.png` shows the three longest non-isolated true
bursts. `short_fragmented_burst_closeups.png` shows length-2-to-4 bursts with
the largest true onset relocations. Both panels select examples from the true
simulated state alone, without looking at fitted-model errors. The short-burst
panel makes the structural failure especially clear: a single no-jump
transition scale either flattens the relocation, smears it over several bins,
or only partially follows it. `tuning_examples.png` shows representative
quantiles of the best no-jump per-neuron tuning correlations rather than only
the worst neurons.

## Why total data creates a sweet spot

Both models use temporal smoothness in the continuous state. The jump model's
advantage is not generic smoothness; it is the ability to switch between a tight
smooth transition and a uniform fragmented transition. A no-jump model must use
one movement scale for both.

The controlled information grid generated one 200-neuron, 8,000-bin source
simulation and fit nested neuron and time prefixes. This removes simulation
seed changes as a confound when varying total data.

| time bins | neurons | jump worst-state MAE | best no-jump MAE | jump tuning | no-jump tuning |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 50 | 0.134 | 0.183 | 0.813 | 0.695 |
| 2,000 | 100 | 0.0822 | 0.152 | 0.862 | 0.682 |
| 2,000 | 200 | 0.00981 | 0.0517 | 0.867 | 0.781 |
| 4,000 | 50 | 0.143 | 0.179 | 0.925 | 0.753 |
| 4,000 | 100 | 0.0543 | 0.115 | 0.934 | 0.842 |
| 4,000 | 200 | 0.0110 | 0.0410 | 0.930 | 0.884 |
| 8,000 | 50 | 0.128 | 0.180 | 0.966 | 0.833 |
| 8,000 | 100 | 0.0541 | 0.111 | 0.965 | 0.895 |
| 8,000 | 200 | 0.00990 | 0.0292 | 0.968 | 0.943 |

Neuron count supplies enough simultaneous evidence for the jump model to decode
each fragmented bin. Duration supplies repeated coverage of the latent space
needed to learn tuning. Too little of either hurts the jump fit. Too much
duration lets the broad no-jump model improve its compromised tuning and partly
catch up. Refining the 200-neuron duration axis located the knee at 3,500 bins:
the jump latent remains essentially exact, tuning is qualitatively recovered,
and the no-jump structural error remains visible.

The duration effect is not perfectly monotone for a single seed. For example,
the 3,000-bin prefix was a poor local fit for both models. The 3,500-bin result
should therefore be treated as a verified simulation regime, not a theorem
that every nearby duration will behave identically.

## Preserved isolated-reset fallback

The earlier isolated-reset candidate remains useful as a stronger fallback. It
uses 100 neurons, 8,000 bins, `p_jump_to_move=0.90`, and rate-bias shift `-1.5`.
Its jump worst-state MAE is `0.0293` and tuning correlation is `0.952`; the best
no-jump fit has MAE `0.101` and tuning correlation `0.872`, a 3.46-fold error
gap. It is retained because it cleanly demonstrates the one-scale tradeoff, but
the preferred default above avoids relying on isolated fragmented bins.

## Reproduce

Activate the established GPU environment and run the preferred defaults:

```bash
source /mnt/home/szheng/miniconda3/etc/profile.d/conda.sh
conda activate jaxnew2
python -u -m poor_man_gplvm.diagnostics.jump_model_separation \
  --output-dir /mnt/home/szheng/ceph/poor_gplvm/diagnostics/jump_model_separation/my_run
```

To reproduce the preserved isolated-reset candidate:

```bash
python -u -m poor_man_gplvm.diagnostics.jump_model_separation \
  --output-dir /mnt/home/szheng/ceph/poor_gplvm/diagnostics/jump_model_separation/isolated_reset_candidate \
  --name isolated_reset_candidate \
  --n-neuron 100 \
  --n-time 8000 \
  --source-n-neuron 100 \
  --source-n-time 8000 \
  --p-jump-to-move 0.90 \
  --rate-bias -1.5
```

The final preferred run is stored remotely at
`/mnt/home/szheng/ceph/poor_gplvm/diagnostics/jump_model_separation/20260723_nonisolated_sweet_spot_final/`
and in the local mirror at
`/Users/madsci/research_figs/ceph/poor_gplvm/diagnostics/jump_model_separation/20260723_nonisolated_sweet_spot_final/`.

On an H100 used during development, the older JAX compiler aborted on a mixed
`float32 posterior @ int32 count` matrix product in the M-step. The diagnostic
stores the same integer-valued counts as `float32`, preserving their values and
allowing the established `jaxnew2` GPU environment to compile the operation. No
optimizer, likelihood, or package environment was changed.
