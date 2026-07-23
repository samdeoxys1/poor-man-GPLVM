# Jump versus no-jump simulation diagnostic

`poor_man_gplvm.diagnostics.jump_model_separation` searches for simulations in
which an explicit jump state remains necessary after granting every fitted
model the strongest simulation-only rescue:

1. the same random posterior initialization;
2. an all-time oracle posterior-overlap permutation;
3. one fixed-posterior Adam M-step after permutation; and
4. one diagnostic E-step with the refitted tuning.

The diagnostic does not initialize any fit from the true latent or the true
parameters. The generative parameter seed is `123`, the fitted parameter seed
is `124`, and the shared posterior seed is `5`.

## Simulation that separates the models

The successful setup has 100 neurons, 8,000 time bins, 100 latent bins,
continuous movement scale `0.35`, tuning lengthscale `5`, and a `-1.5` shift of
the generative bias parameters. Its dynamics transition probabilities are
`p_move_to_jump=0.04` and `p_jump_to_move=0.90`. The observed fragmented
fraction was `0.0459`, so fragmentation usually consists of isolated latent
resets rather than long blocks of independently sampled positions.

After 20 EM iterations and the oracle alignment/refit:

| model | continuous latent MAE | fragmented latent MAE | median tuning correlation |
|---|---:|---:|---:|
| jump | 0.00194 | 0.0293 | 0.952 |
| no jump, scale 0.35 | 0.285 | 0.320 | 0.536 |
| no jump, scale 3 | 0.0748 | 0.191 | 0.867 |
| no jump, scale 10 | 0.0282 | 0.143 | 0.878 |
| no jump, scale 30 | 0.0374 | 0.101 | 0.872 |

Errors are normalized by 100 latent bins. Scale `30` is the best tested
no-jump model by minimax state-specific error, but its worst-state error remains
3.46 times the jump model's error and its tuning is visibly distorted. The jump
model's fragmentation-state accuracy is `0.9908`.

`reset_event_closeups.png` shows the three largest isolated true resets, with
windows selected from the simulated state alone. The jump model relocates in
one bin and then continues smoothly. The strict no-jump fit stays in the old
coordinate, intermediate scales drift gradually to the new coordinate, and the
broadest scale becomes noisy around otherwise stable trajectories.
`tuning_examples.png` shows neurons at representative quantiles of the best
no-jump tuning correlations rather than selecting only the largest failures.

## What made the separation work

The decisive feature was isolated reset events, not simply adding more
fragmented time.

- Long fragmented blocks allow a very broad no-jump transition kernel to behave
  approximately like an independent decoder. With sufficient population
  information, oracle permutation can then recover much of the tuning.
- Reducing firing information indiscriminately does not solve this problem:
  the correct jump model also loses fragmented-state precision because its
  fragmented transition is uniform.
- Isolated resets create a genuine state-dependent transition requirement.
  Most observations benefit from a very tight smooth prior, while a small
  number require a one-bin global relocation. No single no-jump movement scale
  can satisfy both.
- Moderate per-bin information and narrower tuning curves make wrong
  post-reset assignments contaminate the tuning M-step. A global permutation
  cannot repair these segment-specific offsets.

The unsuccessful intermediate regimes were also informative. A simulation with
about 81% fragmented time, 60 neurons, and mean rate `0.655` spikes per neuron
per bin left the jump model's fragmented error at `0.104`; it was too hard for
the correct model. Increasing information and fragmented occupancy improved
jump recovery, but broad no-jump models still recovered tuning correlations
near `0.94`. Distributing comparable population information over 200
lower-rate neurons improved latent recovery but did not create a larger tuning
gap. Changing fragmentation from long blocks to isolated resets produced both
the latent and tuning separation.

## Reproduce

Activate the established GPU environment and run the defaults:

```bash
source /mnt/home/szheng/miniconda3/etc/profile.d/conda.sh
conda activate jaxnew2
python -u -m poor_man_gplvm.diagnostics.jump_model_separation \
  --output-dir /mnt/home/szheng/ceph/poor_gplvm/diagnostics/jump_model_separation/my_run
```

The verified run is stored remotely at
`/mnt/home/szheng/ceph/poor_gplvm/diagnostics/jump_model_separation/jumpsep_stage4_isolated_resets_20260723/`
and in the usual local Ceph mirror at
`/Users/madsci/research_figs/ceph/poor_gplvm/diagnostics/jump_model_separation/jumpsep_stage4_isolated_resets_20260723/`.

On the H100 used for this diagnostic, the older JAX compiler aborted on the
mixed `float32 posterior @ int32 count` matrix product in the M-step. The
diagnostic stores the same integer-valued counts as `float32`, which preserves
their values and lets the established `jaxnew2` GPU environment compile the
operation. No optimizer, likelihood, or package environment was changed.
