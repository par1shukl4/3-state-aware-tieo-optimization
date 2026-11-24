# Analysis of the four corrected TIEO methods on 40 benchmarks

## Data audited

The supplied comparison archive contains:

- 4 methods: baseline, quantile, adaptive z-score and relative deviation;
- 40 benchmark functions;
- 30 paired runs per method and function;
- 4,800 total runs;
- exactly 15,000 objective evaluations in every run;
- identical seed ranges across methods;
- final fitness/error, mean, standard deviation, median, IQR, success rate, Q-tables, state visits, action counts and action successes;
- mean convergence plots for all 40 functions.

The evaluation-budget comparison is therefore fair.

The supplied `paired_statistics_vs_baseline.csv` should not be used directly. Its win/tie/loss counts overlap because a nearly equal result can be counted both as a tie and as a win/loss. It also assigns `p=1` to some small but non-identical values through a broad `allclose` check. The conclusions below were recomputed from `all_methods_runs.csv` using paired seeds and log-error comparisons.

## Executive conclusion

There is no statistically defensible winner among the four state-classification methods.

- Friedman test over the 40 function-level mean errors: `p = 0.338`.
- Pairwise function-level comparisons against baseline:
  - quantile: `p = 0.643`;
  - adaptive z-score: `p = 0.793`;
  - relative deviation: `p = 0.497`.
- After Holm correction across 40 per-function paired tests, none of the three variants records a significant win or loss against baseline.

Quantile has the best numerical average rank, but its advantage is too small and inconsistent to justify replacing the baseline:

| Method | Average rank, lower is better | Mean-error wins | Median-error wins | Lowest-std wins |
|---|---:|---:|---:|---:|
| Quantile | 2.2250 | 13 | 10 | 8 |
| Baseline | 2.4750 | 11 | 15 | 11 |
| Adaptive z-score | 2.5625 | 5 | 7 | 8 |
| Relative deviation | 2.7375 | 11 | 8 | 13 |

The difference between mean and median winners is important. Quantile's rank advantage is partly created by reducing a few large failures, while baseline wins the median on the largest number of functions. Many functions have extremely skewed results, so mean-only ranking is unstable.

Recommendation: retain the two-state baseline as the reference implementation. Do not combine a three-state classifier with further changes yet. The next gains are more likely to come from operator credit, stagnation recovery and late exploitation.

## Category behavior

Average rank by benchmark category:

| Method | Unimodal | Valley | Multimodal | Classic |
|---|---:|---:|---:|---:|
| Baseline | 2.700 | 2.625 | **2.250** | 2.438 |
| Quantile | **2.300** | **1.750** | 2.607 | **1.938** |
| Adaptive z-score | 2.400 | 2.375 | 2.679 | 2.750 |
| Relative deviation | 2.600 | 3.250 | 2.464 | 2.875 |

Interpretation:

- Baseline is the strongest aggregate choice on multimodal functions.
- Quantile is numerically strongest on valley and classic low-dimensional functions.
- Relative deviation produces some excellent individual multimodal results but is inconsistent across the full suite.
- Adaptive z-score does not establish a clear niche and is materially slower in the recorded runs.

## Final-error robustness

Average success rates at error `<= 1e-8` are almost identical:

| Method | Mean success rate over 40 functions | Median success rate |
|---|---:|---:|
| Baseline | 0.6408 | 0.9667 |
| Quantile | 0.6392 | 0.9500 |
| Adaptive z-score | 0.6367 | 0.9333 |
| Relative deviation | 0.6383 | 0.9500 |

These success rates are affected by an incorrect F36 optimum value, discussed below. Because the same metadata was used for all four methods, the comparative conclusion is unchanged.

### Functions solved reliably by all methods

All four methods have median error at or below `1e-8` on 24 functions:

F1–F5, F7, F9–F10, F12–F14, F18–F19, F21, F24, F27–F29, F33–F35, F37, F39 and F40.

Further tuning on these functions has low practical value unless the goal is to reduce evaluations-to-target rather than improve the final value.

### Functions not solved by any method at the median

All methods have median error above `1e-8` on:

- F8 Schwefel 2.22;
- F11 Rosenbrock;
- F15 Bukin N.6;
- F16 Dixon-Price;
- F17 Colville;
- F20 Ackley;
- F22 Schwefel;
- F23 Levy;
- F25 Styblinski-Tang;
- F26 Eggholder;
- F32 Perm;
- F36 Hartmann 3D, although F36 is actually a metadata error rather than an optimizer failure.

These should form the main diagnostic subset for new baseline changes.

### High-variance, basin-dependent functions

Several functions have a good median but a much worse mean because a minority of runs enter the wrong basin:

| Function | Baseline mean error | Baseline median error | Main implication |
|---|---:|---:|---|
| F12 Beale | 2.54e-2 | 8.03e-28 | One failure among otherwise solved runs |
| F26 Eggholder | 3.79e1 | 3.86e-7 | Strong basin-selection problem |
| F29 Schaffer N.4 | 7.49e-5 | 1.57e-12 | Occasional miss dominates mean |
| F30 De Jong N.5 | 2.16 | 1.26e-10 | Roughly half the runs find the best well |
| F37 Hartmann 4D | 1.59e-2 | 2.22e-10 | Rare wrong-basin runs |
| F40 Katsuura | 5.32e-7 | 0 | One late-convergence failure dominates mean |

This is direct evidence for a controlled restart or diversity-recovery mechanism. It is not evidence for finer state division.

## Important per-function observations

### Valley functions

- F11 Rosenbrock: quantile has the lowest mean, 13.64 versus baseline 14.57, but all methods plateau near error 10–30 by approximately 3,000 evaluations. The remaining 12,000 evaluations add almost no improvement.
- F16 Dixon-Price: all medians are approximately 0.6667, showing a common structural limitation.
- F17 Colville: relative deviation has the lowest mean, 0.376, but its median is slightly worse than baseline. It reduces catastrophic failures rather than improving the typical run.
- F15 Bukin: quantile is numerically best, but no method approaches the target tolerance.

These patterns favor decreasing movement size, leader-focused refinement or a small late local search.

### Multimodal functions

- F22 Schwefel: baseline is best and continues improving throughout the full budget. A restart trigger must not activate merely because improvement is slow.
- F25 Styblinski-Tang: baseline/adaptive/quantile continue improving late; relative deviation is clearly worse.
- F26 Eggholder: relative deviation reduces mean error from 37.9 to 14.5, while median performance is near the optimum for baseline and relative deviation. This is primarily a failure-rate problem.
- F30 De Jong N.5: baseline is best, with 53.3% success; three-state methods reduce success to 30–40%.
- F31 Shekel: quantile slightly increases success from 50.0% to 53.3%, but the difference is too small to be conclusive.
- F40 Katsuura: quantile and the other three-state methods make large late improvements after 11,000 evaluations. Baseline has one poor run, but all methods have a zero median error.

### Results that are numerically different but practically tied

Several mean winners differ only below numerical precision, for example Booth, Matyas, Three-Hump Camel, Powell, Griewank, Michalewicz, Branin and Forrester. These should be treated as practical ties even when a rank table assigns positions 1–4.

## Convergence findings

The archive contains mean convergence curves but no convergence standard-deviation bands. Final-value standard deviations are available; time-dependent convergence uncertainty cannot be reconstructed from this zip alone.

The curves show three distinct regimes:

1. **Early plateau:** Rosenbrock, Eggholder and De Jong N.5 make most progress within roughly 1,000–3,000 evaluations and then remain nearly flat.
2. **Continuous slow progress:** Schwefel, Styblinski-Tang and Perm continue improving through much of the 15,000-evaluation budget.
3. **Late discrete breakthroughs:** Katsuura shows large improvements between roughly 11,000 and 15,000 evaluations.

Therefore, use a stagnation trigger based on actual lack of improvement, not a fixed-time restart. Preserve the global best and restart only a small worst fraction.

## What the Q-learning logs reveal

### Actions are almost identical across state methods

| Method | Autonomy share | Diplomacy share | War share |
|---|---:|---:|---:|
| Baseline | 13.32% | 11.72% | 74.96% |
| Quantile | 13.60% | 11.97% | 74.43% |
| Adaptive z-score | 13.70% | 12.00% | 74.30% |
| Relative deviation | 13.38% | 11.89% | 74.73% |

The state classifiers barely change the learned policy. War has the highest mean Q-value in every state and dominates approximately three quarters of all decisions.

This does not automatically mean war should be reduced: war uses only one objective evaluation, while autonomy and diplomacy can evaluate many tribe members. However, the current reward is not normalized by evaluation cost or improvement magnitude, so the Q-values cannot distinguish a tiny cheap success from a large useful improvement.

### Three-state occupancy is unbalanced

| Method | Strong | Neutral | Weak |
|---|---:|---:|---:|
| Quantile | 43.52% | **8.20%** | 48.27% |
| Adaptive z-score | 32.74% | 22.02% | 45.24% |
| Relative deviation | 30.54% | 27.72% | 41.75% |

The quantile neutral state is rarely visited. Tribe collapse frequently leaves only two active tribes, causing the fallback strong/weak rule. The added Q-table row therefore contributes little useful specialization.

## Runtime findings

Median runtime per run:

| Method | Median seconds |
|---|---:|
| Baseline | 1.052 |
| Relative deviation | 1.077 |
| Quantile | 1.129 |
| Adaptive z-score | 2.757 |

The mean runtimes are corrupted by a few extreme stalls, including runs recorded at 8,172 seconds, 6,068 seconds and 1,218 seconds despite normal iteration counts. These are likely environment/resource interruptions rather than algorithmic work. Use median runtime and profile a clean local sample before making speed claims.

## Benchmark metadata corrections required

### F36 Hartmann 3D

The implemented coefficients have a minimum of approximately:

`-3.862779787332663`

The registry uses `-3.86278214782076`, creating an artificial irreducible error of `2.360488e-6`. All four algorithms are reaching the implemented minimum; F36 should be considered solved after correcting the registry.

### F25 Styblinski-Tang

The accurate minimum is approximately:

`-39.16616570377142 × dimension`

The registry uses `-39.16599 × dimension`. At dimension 30 this shifts the target by approximately `0.00527`. It does not change the method ordering materially, but it should be fixed before future reports.

## Prioritized minute changes for the baseline

### Priority 1 — Evaluation-aware, magnitude-aware Q reward

Evidence:

- War consumes about 75% of actions in every method.
- All state classifiers lead to almost the same policy.
- Current rewards are mostly binary and do not represent improvement magnitude or evaluation cost.

Test this single change first:

```text
scale = max(abs(old_stat), population_MAD, epsilon)
raw_improvement = (old_stat - new_stat) / scale
reward = clip(raw_improvement, -1, 1) / evaluations_used
```

Use actual war outcome rather than a state-dependent fixed reward. Continue logging action share, success rate and improvement per evaluation.

Hypothesis: diplomacy/autonomy will be selected when their larger multi-member changes justify their cost, while war will remain favored only when it is genuinely efficient.

### Priority 2 — Stagnation-triggered partial restart

Evidence:

- Beale, Eggholder, De Jong, Schaffer N.4, Hartmann 4D/6D and Katsuura show rare catastrophic misses.
- Several convergence curves plateau for most of the remaining budget.

Minimal experiment:

- trigger after 1,500 evaluations without meaningful global-best improvement;
- restart the worst 5% or at least one individual;
- preserve the global best;
- allow at most two restarts;
- reset or partially decay the affected tribe's Q history only if necessary.

Compare patience values 1,500 and 3,000 on F12, F26, F29, F30, F31, F37, F38 and F40.

### Priority 3 — Decreasing autonomy weight

Evidence:

- Rosenbrock, Dixon-Price, Colville and Bukin remain far from optimum.
- Their curves plateau after rapid initial improvement.

Test only two schedules against constant `w=0.5`:

```text
linear:    w = 0.9 - 0.75 * progress
quadratic: w = 0.15 + 0.75 * (1 - progress)^2
```

This supplies large early motion and finer late exploitation without changing the operator formula.

### Priority 4 — Small late local refinement

If changing `w` is insufficient, reserve the final 5% of the budget for a bounded coordinate search around the external best. Test primarily on F11, F15, F16, F17 and F32.

This should be reported as a hybrid TIEO variant because it adds a new search component.

### Priority 5 — Reflective boundary handling

Compare clipping with reflection on F15 Bukin, F22 Schwefel and F26 Eggholder. Do not promote it based on easy bowl functions. Eggholder has an optimum on a boundary, so verify that reflection does not prevent evaluating boundary points.

### Priority 6 — Soft elite diplomacy target

This is lower priority than reward and stagnation fixes. If tested, change only diplomacy target selection:

```text
p(target) proportional to exp(-rank / temperature)
```

Keep a nonzero chance of selecting non-best tribes. Test on F22, F25, F26, F30, F31 and F38.

## Changes not recommended yet

- Do not add more fitness states. The present three-state methods do not materially alter policy or final reliability.
- Do not combine quantile, elite ranking, restarts and new rewards in one experiment.
- Do not reduce the evaluation budget globally based only on early plateaus; Schwefel, Styblinski-Tang, Perm and Katsuura continue improving late.
- Do not use mean error alone. Report median, IQR, success rate and mean error together.
- Do not use the supplied paired-statistics CSV without correcting its tie and `allclose` logic.

## Efficient experiment sequence

Use the corrected two-state baseline and isolate one change at a time:

| Experiment | Change | Diagnostic functions |
|---|---|---|
| B0 | Corrected baseline | Reference |
| B1 | Magnitude/evaluation-aware reward | F8, F11, F17, F22, F26, F30, F32, F40 |
| B2 | Partial restart only | F12, F26, F29, F30, F31, F37, F38, F40 |
| B3 | Linear `w` schedule only | F11, F15, F16, F17, F32 |
| B4 | Quadratic `w` schedule only | Same as B3 |
| B5 | Reflective bounds only | F15, F22, F26 |
| B6 | Soft elite diplomacy only | F22, F25, F26, F30, F31, F38 |

For rapid development, one run is acceptable as a smoke test. It is not enough to decide between changes on basin-dependent functions. After a change passes the one-run screen, use at least 10 paired seeds on its diagnostic subset. The current 30-run data demonstrate why: a single Eggholder, De Jong, Shekel or Hartmann run can give the opposite conclusion from the full distribution.

## Final decision

Keep the corrected baseline as the master reference. The first new optimization file should implement evaluation-aware normalized reward. The second should implement a small stagnation-triggered partial restart. Only after those isolated tests should the best two changes be combined.
