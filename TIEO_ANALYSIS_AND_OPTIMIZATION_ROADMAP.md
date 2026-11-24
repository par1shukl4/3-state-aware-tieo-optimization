# TIEO code and results analysis

## Executive conclusion

The attached experiments do **not** show that a three-state TIEO is intrinsically worse than the two-state baseline. The three-state notebooks change several other parts of the optimizer at the same time, including the operators, acceptance rule, clustering schedule, action semantics, target selection, reward calculation, and in some cases the benchmark functions. The current experiments therefore cannot isolate the effect of state division.

From the embedded results, the baseline has the best reported mean result on all 16 functions against every completed variant. Among the non-baseline completed variants, quantile division is the most promising: it is the best non-baseline method on 12 of 16 functions and has an average rank of 2.31 across the five completed methods. This is evidence for testing quantile states again in a controlled ablation, not evidence that the present quantile implementation beats the baseline.

The first optimization step should be to build one canonical baseline implementation and make states, rewards, target selection, and optional enhancements interchangeable components. Until that is done, further combined variants will be hard to interpret.

## What is in the archive

| Notebook | Intended change | Execution status in attached output |
|---|---|---|
| `Baseline_TIEO.ipynb` | Two states | Completed, 16 functions x 30 runs |
| `Adaptive_Zscore_TIEO.ipynb` | Three states using z-score threshold | Completed, 16 x 30 |
| `Quantile_TIEO.ipynb` | Three states using 33.33/66.67 percentiles | Completed, 16 x 30 |
| `Relative_Deviation.ipynb` | Three states using relative deviation | Incomplete; F1 and F2 complete, F3 stops after run 6 |
| `Global_Ranking_TIEO.ipynb` | Elite-rank target selection | Completed, 16 x 30 |
| `Adaptive+Elite.ipynb` | Adaptive z-score plus elite selection | Completed, 16 x 30 |
| `Quantile+Elite.ipynb` | Quantile plus elite selection | Failed during F1 with an `IndexError` |

## Reported performance

The table below uses the reported mean final objective. For F13 it shows mean error from the stated optimum, -12569.5. Lower is better. F15 and F16 are marked because their mathematical definitions are not consistent across notebooks.

| Function | Baseline | Adaptive-Z | Quantile | Elite rank | Adaptive-Z + elite |
|---|---:|---:|---:|---:|---:|
| F1 Sphere | 1.844e-10 | 2.121e+02 | 4.317e+00 | 6.553e+03 | 3.681e+03 |
| F2 Schwefel 2.22 | 4.045e-05 | 4.950e+00 | 2.135e+00 | 3.425e+01 | 1.832e+01 |
| F3 Schwefel 1.2 | 6.325e-06 | 8.922e+02 | 1.119e+03 | 2.436e+04 | 8.253e+03 |
| F4 Schwefel 2.21 | 2.278e-05 | 8.435e+00 | 1.052e+01 | 5.122e+01 | 2.269e+01 |
| F5 Shifted sphere | 1.067e-02 | 2.636e+02 | 4.026e+00 | 6.163e+03 | 3.629e+03 |
| F6 Rosenbrock | 5.627e+00 | 4.205e+04 | 8.697e+02 | 5.243e+03 | 4.632e+05 |
| F7 Noisy quartic | 6.090e-03 | 3.548e-01 | 4.048e-01 | 1.649e+01 | 1.617e-01 |
| F8 Weighted power | 3.243e-15 | 1.000e-06 | 1.033e-09 | 9.182e-02 | 1.392e-05 |
| F9 Custom | 4.527e-10 | 5.064e-01 | 3.434e-02 | 3.989e+00 | 1.348e+00 |
| F10 Weighted quartic | 2.688e-18 | 5.010e+04 | 1.819e+03 | 7.237e+07 | 9.131e+07 |
| F11 Rastrigin | 6.794e-07 | 9.512e+01 | 5.755e+01 | 2.671e+02 | 6.644e+01 |
| F12 Griewank | 3.173e-11 | 1.075e+00 | 8.460e-01 | 2.347e+02 | 4.051e+01 |
| F13 Schwefel 2.26, error | 3.342e+03 | 1.946e+04 | 1.836e+04 | 1.775e+04 | 2.154e+04 |
| F14 Ackley | 5.177e-06 | 5.104e+00 | 2.941e+00 | 1.854e+01 | 1.133e+01 |
| F15 Penalised 1* | 1.970e-03 | 1.323e+01 | 2.557e+00 | 3.790e+01 | 1.412e+02 |
| F16 Penalised 2* | 5.623e-04 | 1.131e+02 | 1.493e+00 | 2.964e+01 | 1.234e+03 |

Average ranks over these reported means are: baseline 1.00, quantile 2.31, adaptive-Z 3.13, adaptive-Z plus elite 4.13, and elite rank 4.44. The baseline is also much faster: the median runtime multipliers relative to it are approximately 5.22x for adaptive-Z, 6.19x for quantile, 1.19x for elite rank, and 2.39x for adaptive-Z plus elite.

These ranks are descriptive only. A formal significance test should be performed from saved per-run data after the benchmark definitions and evaluation budgets are made identical.

## Why the current variants regress

### 1. The three-state notebooks remove greedy acceptance

The baseline evaluates candidate positions and retains a candidate only when it improves the individual's fitness. The adaptive-Z, quantile, and relative-deviation implementations overwrite all members with the new positions and fitness values, including worsening moves. This is likely the single largest reason they regress by orders of magnitude.

Fix: make the baseline acceptance rule common to every variant. If occasional worsening moves are desired, add them as a separate, explicitly tested mechanism such as simulated-annealing acceptance.

### 2. They are not state-only ablations

The three-state notebooks recreate tribes with K-means on every iteration. The baseline creates tribes initially and preserves membership except for war/collapse. Their diplomacy and war operators also differ from the baseline. Therefore the comparison changes far more than two states versus three states.

Fix: copy the baseline implementation exactly and replace only `get_state()` with a three-state classifier and resize `Q` from `(2, 3)` to `(3, 3)`.

### 3. Selected Q-actions and executed operators do not always match

Examples:

- Adaptive-Z: when the state is neutral, the selected Q-action is ignored and autonomy or diplomacy is chosen by a fresh coin flip; the Q-value for the ignored action is still updated.
- Quantile: in the strong state, actions 0 and 1 both execute autonomy; in the weak state, actions 0 and 1 both execute diplomacy.
- Relative deviation: the weak state similarly maps both actions 0 and 1 to diplomacy.

This makes the learned `Q[state, action]` value causally incorrect.

Fix: action 0 must always mean autonomy, action 1 diplomacy, and action 2 war. If state-dependent action sets are desired, mask unavailable actions before selection and update the action that was actually executed.

### 4. Diplomacy no longer means inter-tribe diplomacy

In the three-state notebooks, the diplomatic leader is a randomly selected member of the same tribe. The baseline uses another tribe's leader. This removes the cross-tribe information flow that diplomacy is intended to provide.

Fix: restore cross-tribe targeting. Test elite-biased target selection only after this behavior is restored.

### 5. Quantile + elite currently contains a call-site bug

`run()` passes the full `countries` list as the `members` argument to `diplomacy()` instead of `countries[c]`. The attached run fails in F1 with a boolean-index shape mismatch.

Fix: pass the current country's member list. Add a one-run smoke test for every action before starting a 30-run benchmark.

### 6. Randomness is not fully controlled

F7 calls `np.random.uniform()` rather than the optimizer's seeded generator. Most benchmark loops do not seed global NumPy state. `Adaptive+Elite.autonomy()` also calls `np.random.random()` instead of `self.rng.random()`. Thus the same listed seed does not fully reproduce a run.

Fix: pass one `numpy.random.Generator` everywhere. For a noisy objective, either use common random numbers across algorithms or evaluate with a defined resampling protocol.

### 7. Benchmark functions are inconsistent

F15 and F16 differ materially between notebook groups. For example, quantile/relative/global use the same simplified expression for both F15 and F16, adaptive-Z uses another F15 expression, and combined variants use yet another pair. These are not comparable results. The baseline benchmark dictionary and most benchmark definitions are not present in the saved source cell even though its output exists, so the baseline notebook is not reproducible from a clean kernel.

Fix: define benchmarks once in a shared module and import the exact same function objects into every experiment. Add unit tests at the known optimum and at several fixed vectors.

### 8. The iteration budget is not a fair evaluation budget

Autonomy/diplomacy evaluate an entire tribe; war evaluates one point. The chosen action mix therefore changes the number of objective evaluations. The baseline also calls `self.f()` directly in war without incrementing `eval_count`. Algorithms are currently compared at 300 iterations, not at an identical number of function evaluations.

Fix: enforce a hard function-evaluation budget and route every objective call through one counted `evaluate()` method.

### 9. Elite ranking changed the operators too

The global-ranking implementation replaces baseline autonomy with small isotropic noise and diplomacy with copying 80% of an elite leader's coordinates. Its poor result therefore does not isolate elite target selection. Adaptive-Z plus elite is another substantial rewrite with a different exploration schedule and operators.

Fix: on the unchanged baseline, alter only the probability of selecting a diplomacy target. Compare uniform target selection with rank-biased selection.

## Prioritized optimization roadmap

### Stage 0 — make the evidence trustworthy

1. Create one canonical optimizer class and one benchmark module.
2. Route every objective evaluation through a counted wrapper.
3. Use a strict budget, for example 15,000 evaluations, rather than 300 iterations.
4. Use one seeded RNG; remove all direct global `np.random` calls.
5. Restore greedy acceptance for all three operators.
6. Keep action meanings identical in every state.
7. Save per-run final error, evaluation count, convergence by evaluation count, state occupancy, action counts, acceptance counts, and final Q-table.
8. Add smoke tests for each state/action and optimum-value tests for all 16 functions.

This stage may produce a larger gain in reliability than any new optimization mechanism.

### Stage 1 — minute, low-risk baseline improvements

Test each item independently:

1. **Count war evaluations correctly.** This is a correctness fix.
2. **Do not re-evaluate an unchanged leader in autonomy.** The leader's variance is effectively zero and the evaluation is usually wasted.
3. **Vectorize improvement assignment.** Replace Python member loops with indexed array updates; this improves runtime without changing behavior.
4. **Use reflection rather than clipping at bounds.** Clipping can accumulate coordinates exactly on a boundary and reduce diversity.
5. **Randomize ties with the seeded RNG.** Avoid deterministic `argmax` preference for action 0.
6. **Scale coordinates before K-means** when dimensions have different bounds in future problems.
7. **Record state occupancy.** With only 2–5 tribes, a three-way state rule can leave a state almost unused.
8. **Cache tribe statistics** within an action step rather than recomputing all state statistics repeatedly.

### Stage 2 — improve learning without changing operators

1. Replace binary reward with a scale-free signed improvement:

   `reward = clip(log((old + eps)/(new + eps)), -r_max, r_max)`

   For objectives that may be negative, use improvement divided by a robust scale such as `abs(old) + median_abs_deviation + eps`.

2. Divide reward by the number of evaluations used by the action. This teaches efficiency, not just success probability.
3. Give war an outcome-based reward; do not assign `0.1` merely because the pre-action state was strong.
4. Test `gamma` in `{0, 0.5, 0.9}`. Because actions immediately alter the population, a lower discount may make credit assignment cleaner.
5. Replace the very large initial learning rate with a visit-count schedule such as `alpha = 1 / sqrt(N(s,a))`, clipped to `[0.05, 0.5]`.
6. Test an exploration floor of 0.05–0.10 and log action entropy. Premature Q dominance is easy with only non-negative rewards.
7. Consider Double Q-learning only after the reward and action-accounting issues are fixed.

### Stage 3 — clean two-state versus three-state ablation

Use the unchanged baseline operators and compare only:

- Two-state mean split, current baseline.
- Three-state rank/quantile split.
- Three-state robust z-score split using median and MAD rather than mean and standard deviation.
- Three-state temporal split based on improvement rate or stagnation, not just absolute fitness.

The first three-state candidate should be rank based. With at most five tribes it is scale invariant and more stable than relative deviation. Suggested rule:

- strongest `ceil(k/3)` tribes: strong;
- weakest `ceil(k/3)` tribes: weak;
- remaining tribes: neutral;
- when `k < 3`, fall back to the original two states.

Add hysteresis: require a tribe to cross a boundary for two consecutive decisions before changing state. This avoids Q-table thrashing when tribe means are close.

The current results make quantile division the best candidate to revisit, but use the clean rule above rather than the current rewritten optimizer.

### Stage 4 — operator improvements

1. **Time-varying autonomy weight:** test `w(t)` decreasing from about 0.8–1.0 to 0.1–0.2.
2. **Diversity-adaptive weight:** increase step size when normalized population diversity falls below a threshold.
3. **Leader-directed diplomacy:** use a bounded differential move such as `x + F*(elite-x) + F2*(r1-r2)` and retain greedy selection.
4. **Elite archive:** keep a small external archive of distinct good solutions; do not use a single global elite exclusively.
5. **Stagnation restart:** reinitialize only the worst 5–10% after a defined number of evaluations without global improvement.
6. **Late local search:** apply a small coordinate or quasi-Newton search to the best solution only in the final 10–20% of the budget.
7. **Diversity-preserving war:** transfer or regenerate the worst member only if it does not duplicate the winner closely.

### Stage 5 — elite ranking, isolated correctly

Keep baseline operators and change only diplomacy target selection:

- Uniform target, baseline.
- Inverse-rank probability, `p_i proportional to 1/rank_i`.
- Softmax rank with temperature that decreases over time.
- Top-pool sampling, selecting uniformly from the top 20–40%.

Avoid always selecting the best tribe; it can cause premature convergence. A soft distribution over several elites is safer. Elite ranking should be combined with a three-state method only after both independently beat or match the baseline.

## Recommended experiment sequence

Use a small screening set first: Sphere, Rosenbrock, Rastrigin, Schwefel 2.26, and noisy Quartic.

| Experiment | Single change from canonical baseline | Purpose |
|---|---|---|
| A0 | None | Reproduced reference |
| A1 | Correct evaluation counting + hard FE budget | Fair reference |
| A2 | Signed normalized reward | Better credit assignment |
| A3 | Reflective boundary handling | Reduce boundary collapse |
| A4 | Decreasing `w(t)` | Exploration/exploitation balance |
| B1 | Three-state rank/quantile classifier only | State ablation |
| B2 | Robust median/MAD classifier only | Outlier-resistant state ablation |
| C1 | Inverse-rank diplomacy target only | Elite-selection ablation |
| C2 | Softmax-rank target only | Reduce elite over-exploitation |
| D1 | Best independently validated state + target changes | Interaction test |

Screen with 10 paired seeds. Promote only changes with a useful effect size and no catastrophic failures. Then run 30 paired seeds on all 16 functions.

For the final analysis, report median and IQR in addition to mean and standard deviation; convergence versus function evaluations; success rate at a predeclared tolerance; paired Wilcoxon tests per function with Holm correction; and a Friedman test with average ranks across functions. Also report an effect size such as paired Vargha–Delaney A12 or rank-biserial correlation.

## Immediate recommendation

Do not tune the current three-state thresholds yet. First produce a clean baseline-derived three-state implementation with identical operators, greedy acceptance, identical benchmarks, one RNG, and a hard evaluation budget. Test rank/quantile states first. In parallel, correct the baseline's evaluation counting and reward definition. Those changes give the fastest path to a defensible optimization result.
