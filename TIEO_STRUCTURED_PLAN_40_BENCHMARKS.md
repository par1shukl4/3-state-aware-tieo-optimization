# Structured TIEO correction and optimization plan

## 1. Objective

The work should answer three separate questions in order:

1. Can the existing baseline and variants be reproduced fairly?
2. Does a specific change improve the baseline when nothing else changes?
3. Does the final improved TIEO generalize across 40 benchmarks rather than only the current 16?

The current archive answers none of these conclusively because the variants change multiple mechanisms simultaneously. The plan below first creates a reliable reference implementation, then tests small changes individually, and finally evaluates only the strongest combinations on all 40 functions.

## 2. Important limitation of the current evidence

The attached notebooks contain complete results for 16 functions only. Therefore, no claim can yet be made about actual performance on 40 benchmarks.

Current measured conclusion:

- The reported baseline mean is best on all 16 attached functions.
- Quantile is the best non-baseline completed variant on 12 of the 16 functions.
- This is not a clean two-state versus three-state comparison because the operators, acceptance rules, clustering, reward logic, and some benchmark definitions differ.
- `Relative_Deviation` is incomplete.
- `Quantile+Elite` fails during F1.

Performance on 40 functions must be measured after the corrections in this plan.

---

# Part A — Fix the existing code

## Phase A0 — Preserve the current work

Before changing algorithm behavior:

- Keep the original notebooks read-only as historical records.
- Create a new canonical implementation rather than continuing to copy and modify entire notebooks.
- Export the existing embedded 30-run results into a separate archival CSV if possible.
- Record the source notebook, algorithm name, date, parameters, and known issues for every historical result.

Deliverables:

- `legacy/` containing the original notebooks.
- `src/tieo.py` containing one optimizer implementation.
- `src/benchmarks.py` containing all benchmark definitions.
- `experiments/` containing configurations rather than duplicate optimizer code.
- `tests/` containing correctness and reproducibility checks.

## Phase A1 — Create one canonical TIEO implementation

Do not maintain separate full implementations for baseline, quantile, z-score, relative deviation, and elite ranking. Use one class with interchangeable components:

```text
TIEO
 ├── state_classifier
 │    ├── mean_two_state
 │    ├── rank_three_state
 │    ├── robust_z_three_state
 │    └── relative_deviation_three_state
 ├── reward_model
 ├── target_selector
 │    ├── uniform
 │    └── elite_ranked
 ├── boundary_handler
 └── movement_schedule
```

All variants must use the same initialization, tribe representation, operators, acceptance rule, evaluation counter, stopping condition, and result logger.

Acceptance criterion:

- Switching the state classifier off or selecting `mean_two_state` must reproduce the corrected baseline.
- A state-classifier experiment should require changing only a configuration value.

## Phase A2 — Centralize and verify benchmark functions

This is mandatory before further benchmarking.

Problems to fix:

- F15 and F16 are defined differently across the notebooks.
- Some notebooks use simplified or duplicated penalized functions.
- The baseline notebook's saved source does not contain the full benchmark dictionary needed to reproduce its embedded output from a clean kernel.
- F7 uses global random noise.

Actions:

1. Define every benchmark once in `benchmarks.py`.
2. Store for each function:
   - name and identifier;
   - dimension;
   - lower and upper bounds;
   - known optimum location when available;
   - known optimum value;
   - whether it is deterministic, noisy, shifted, rotated, or constrained.
3. Add tests that evaluate every function at its known optimum.
4. Add fixed-vector regression tests so accidental formula changes are detected.
5. Decide whether F7 is intentionally noisy:
   - if no, remove the random term;
   - if yes, pass a controlled RNG and use a defined repeated-evaluation protocol.
6. Use the same function object and optimum metadata for every algorithm.

Acceptance criterion:

- All algorithms produce identical objective values for the same 40 fixed input vectors.

## Phase A3 — Fix objective-evaluation accounting

Current problem:

- Autonomy and diplomacy evaluate many individuals, while war often evaluates one.
- Baseline war calls the objective directly without incrementing `eval_count`.
- Comparing methods after 300 iterations does not give them identical computational budgets.

Actions:

1. Route every objective call through `evaluate()`.
2. Increment the counter by the number of candidate vectors evaluated.
3. Stop before the next operation would exceed the hard evaluation budget, or evaluate only the remaining allowed candidates.
4. Store convergence against evaluation count, not only iteration.
5. Use runtime as a secondary metric, not as the primary stopping rule.

Recommended primary budget for the first 30-dimensional study:

- 15,000 function evaluations, approximately 500 evaluations per dimension.
- Save checkpoints at 3,000, 7,500, and 15,000 evaluations.
- If an official benchmark suite requires another budget, follow that suite's official protocol and report it separately.

Acceptance criterion:

- Every compared run ends with the same evaluation budget.
- Reported evaluation count exactly matches an independent test counter.

## Phase A4 — Restore greedy acceptance consistently

Current problem:

- The baseline accepts only improvements.
- The three-state notebooks overwrite positions and fitness values even when candidates are worse.

Actions:

```python
candidate_f = evaluate(candidate_x)
improved = candidate_f < current_f
X[improved] = candidate_x[improved]
F[improved] = candidate_f[improved]
```

Apply the same rule to autonomy and diplomacy in every variant. For war, explicitly decide and document whether replacement is mandatory or greedy. The first fair comparison should preserve baseline war behavior.

If accepting worse solutions is later tested, treat it as a separate named experiment with an explicit probability or temperature schedule.

Acceptance criterion:

- With greedy mode enabled, no accepted individual fitness becomes worse after autonomy or diplomacy.

## Phase A5 — Make selected actions equal executed actions

Required invariant:

| Q-table action | Operator |
|---:|---|
| 0 | Autonomy |
| 1 | Diplomacy |
| 2 | War |

Fix these existing issues:

- Adaptive-Z must not ignore the chosen action in the neutral state.
- Quantile must not map both strong-state actions 0 and 1 to autonomy.
- Quantile and relative deviation must not map both weak-state actions 0 and 1 to diplomacy.
- The Q-value updated must correspond to the operator actually executed.

If an action is unavailable, mask it before selection. Do not silently substitute another action.

Acceptance criterion:

- A test with mocked operators confirms that every selected action invokes exactly one matching operator and updates the matching Q-table cell.

## Phase A6 — Restore correct diplomacy behavior

Current problem:

- Some variants select a random member of the same tribe as the diplomatic target.
- Baseline diplomacy uses another tribe's leader.

Actions:

1. Restore cross-tribe diplomacy.
2. Exclude the acting tribe from the target set.
3. Handle the one-tribe case by selecting autonomy before the action is logged.
4. Use uniform target selection for the corrected baseline.
5. Add elite-biased selection only as a later, isolated experiment.

Acceptance criterion:

- Every diplomacy target belongs to a different active tribe.

## Phase A7 — Fix `Quantile+Elite`

The call to `diplomacy()` passes the complete `countries` list as the member list. It should pass the current country members, equivalent to `countries[c]`.

After correcting it:

- Run one deterministic smoke test forcing action 0.
- Run one forcing action 1.
- Run one forcing action 2.
- Run a complete short optimization with population 10, dimension 5, and 100 evaluations.
- Do not start the 40-function experiment until all four tests pass.

## Phase A8 — Use one random-number generator

Actions:

- Use `self.rng = np.random.default_rng(seed)`.
- Replace every direct `np.random.*` call in algorithm code with `self.rng.*`.
- Give noisy benchmark functions their own explicitly seeded generator.
- Use the same seed list for all compared configurations.
- Store the seed in every result row.

Acceptance criterion:

- Two runs with identical configurations and seeds produce byte-equivalent numeric result files.

## Phase A9 — Remove clustering as a hidden confound

The baseline preserves tribes while the three-state notebooks re-run K-means each iteration. These must not be compared as state-only variants.

Corrected baseline behavior:

- Initialize tribes once.
- Update tribe statistics after actions.
- Change membership only according to the defined war/collapse mechanism.

Later separate experiment:

- Periodic reclustering every `R` evaluations.

Never combine periodic reclustering with a new state classifier in the first test.

## Phase A10 — Protect the global best

Actions:

- Maintain an external copy of the best solution and fitness found so far.
- Never allow bookkeeping or tribe membership changes to lose it.
- Optionally reinsert it only in a separately named elitism experiment; recording it externally is sufficient for the baseline.

Acceptance criterion:

- The recorded best-so-far convergence curve is monotonically non-increasing for minimization.

## Phase A11 — Make results auditable

Save one row per run with:

- algorithm/configuration identifier;
- benchmark identifier and dimension;
- seed;
- final best objective and error from optimum;
- median population fitness;
- evaluation count and runtime;
- success/failure status;
- final Q-table;
- number of visits to each state;
- number of selections and successful improvements for each action;
- acceptance rate per action;
- diversity checkpoints;
- convergence at fixed evaluation checkpoints.

Do not rely only on notebook output cells.

---

# Part B — Minute optimization changes

Each change below must first be tested alone against the corrected baseline. The priority combines expected value, implementation effort, and interpretability.

## Priority 1 — Very small correctness or efficiency changes

| ID | Change | Exact experiment | Expected effect | Risk |
|---|---|---|---|---|
| M01 | Avoid evaluating unchanged leaders | Skip the tribe leader during autonomy when its candidate is numerically identical | More useful evaluations under a fixed budget | Very low |
| M02 | Seeded random tie-breaking | Randomly choose among equal maximum Q-values using `self.rng` | Removes permanent bias toward action 0 | Very low |
| M03 | Reflective boundaries | Compare clipping with repeated reflection into `[lb, ub]` | Less boundary accumulation and better diversity | Low |
| M04 | Cache tribe statistics | Recompute only after an operation changes members/fitness | Lower runtime without behavioral change | Very low |
| M05 | Vectorize accepted updates | Replace member update loops with indexed masks | Lower runtime without changing results | Very low |
| M06 | Scale K-means inputs | Normalize each coordinate using its search range | Better tribes when dimensions have different ranges | Low |
| M07 | Stable numerical comparisons | Use explicit tolerances for Q ties and improvement | Avoids floating-point action instability | Very low |

## Priority 2 — Small learning improvements

### M08 — Magnitude-aware reward

Binary reward treats a tiny and a large improvement equally. Test:

```text
scale = max(abs(old_stat), robust_population_scale, epsilon)
reward = clip((old_stat - new_stat) / scale, -1, 1)
```

Use tribe mean or median consistently. For noisy benchmarks, use a small tolerance before declaring an improvement.

Compare:

- binary reward, corrected baseline;
- normalized signed improvement;
- normalized improvement per objective evaluation.

### M09 — Outcome-based war reward

Remove the fixed reward based only on whether the acting tribe was strong. Reward war using the actual change in loser/winner statistics or global best, normalized by evaluation cost.

### M10 — Visit-count learning rate

Compare the iteration-based learning rate with:

```text
alpha(s,a) = clip(1 / sqrt(visits(s,a)), 0.05, 0.5)
```

This responds to how often a state-action pair is actually observed.

### M11 — Lower discount factor

Test `gamma` in `{0.0, 0.5, 0.9}`. Because the effect of an operator is observed immediately, `gamma=0.9` may overvalue bootstrapped estimates.

### M12 — Exploration floor

Test final exploration probabilities `{0.05, 0.10, 0.15}` while keeping the initial probability fixed. Log action entropy so apparent gains can be linked to behavior.

## Priority 3 — Small movement changes

### M13 — Decreasing autonomy weight

Compare constant `w=0.5` with:

```text
w(t) = w_min + (w_max - w_min) * (1 - progress)^p
```

Initial grid:

- `w_max` in `{0.8, 1.0}`;
- `w_min` in `{0.1, 0.2}`;
- `p` in `{1, 2}`.

Screen only four representative combinations rather than a full grid.

### M14 — Diversity-adaptive weight

Increase `w` when normalized population diversity falls below a threshold and reduce it near the end of the budget. Test only after M13.

### M15 — Better diplomacy interpolation

Instead of copying coordinates directly, test:

```text
candidate = x + beta * (target_leader - x)
```

with `beta` sampled from `[0, 1]`, followed by a small differential perturbation. Preserve greedy acceptance.

### M16 — Coordinate mask probability

Test diplomacy mask probabilities `{0.3, 0.5, 0.8}`. Large masks may cause rapid loss of diversity.

### M17 — Scale-aware war perturbation

Replace a fixed normal standard deviation with a value based on winner-loser distance and remaining search progress. Cap it by a fraction of the variable range.

## Priority 4 — Three-state classifiers, tested cleanly

### M18 — Rank/quantile three-state classifier

Recommended first classifier:

- If `k < 3`, use the original two-state rule.
- Otherwise rank active tribes by their average or median fitness.
- Best `ceil(k/3)` tribes are strong.
- Worst `ceil(k/3)` tribes are weak.
- Remaining tribes are neutral.

Advantages:

- scale invariant;
- unaffected by negative objectives;
- robust when function magnitudes change during convergence;
- guarantees meaningful occupancy when enough tribes exist.

### M19 — Robust z-score classifier

Use median and median absolute deviation:

```text
z_robust = (tribe_stat - median) / (1.4826 * MAD + epsilon)
```

Test thresholds `{0.5, 1.0}`. Fall back to ranks when MAD is nearly zero.

### M20 — State hysteresis

Require a proposed state change to persist for two consecutive decisions before applying it. This may reduce Q-table thrashing near a threshold.

### M21 — Progress-aware states

Add recent improvement rate or stagnation as a state signal only after fitness-based three-state experiments are understood. Do not immediately expand to a large state space; begin with three states such as improving, stable, and stagnating.

## Priority 5 — Elite mechanisms

### M22 — Inverse-rank diplomatic targets

Keep every baseline mechanism unchanged and use:

```text
p(target_i) proportional to 1 / rank_i
```

This must affect diplomacy target selection only.

### M23 — Softmax-rank targets

Test:

```text
p(target_i) proportional to exp(-rank_i / temperature)
```

Use a high temperature early and lower temperature late. This is less aggressive than always using the best tribe.

### M24 — Top-pool sampling

Select uniformly from the top `{20%, 40%}` of eligible tribes. This is simple and may preserve more diversity than inverse-rank selection.

### M25 — Small diverse elite archive

Maintain 3–5 good solutions separated by a minimum normalized distance. Sample diplomatic targets from the archive. Test only after M22–M24.

## Priority 6 — Controlled diversity recovery

### M26 — Stagnation-triggered partial restart

When the global best has not improved for a fixed number of evaluations, reinitialize only the worst 5–10% of individuals. Preserve the external best.

Test trigger lengths of `{0.10, 0.20}` times the total evaluation budget.

### M27 — Opposition-based replacement

For a restarted individual, compare a random point and its opposite point relative to the search-range midpoint, retaining the better. Count both evaluations.

### M28 — Late local search

Spend at most 5–10% of the evaluation budget on the best solution during the final stage. Treat this as a hybrid algorithm and report it separately from pure TIEO.

---

# Part C — Forty-benchmark evaluation plan

## C1. Freeze the exact suite before running experiments

The exact 40 functions are not present in the attachment. Before implementation, create a manifest listing all 40 functions and their sources. Do not silently construct a new suite while experiments are running.

If choosing a custom 40-function suite, use a balanced structure such as:

| Category | Suggested count | Purpose |
|---|---:|---|
| Unimodal/separable | 6 | Exploitation and convergence speed |
| Unimodal/non-separable or ill-conditioned | 6 | Directional search and conditioning |
| Basic multimodal | 10 | Escaping local minima |
| Shifted/rotated/noisy | 8 | Robustness and coordinate dependence |
| Hybrid/composition | 10 | Generalization and mixed landscapes |
| Total | 40 | |

Alternatively, use a recognized official benchmark suite and follow its official functions, transformations, optima, dimensions, and evaluation budget exactly.

Do not mix functions from incompatible suites without documenting every formula and transformation.

## C2. Experimental constants

Primary comparison:

- Dimension: 30.
- Population: 50 unless population size itself is the tested change.
- Runs: 10 paired independent seeds per function and configuration for routine comparison. Use 30 only for the final confirmation of the selected methods.
- Seeds: one frozen list shared by every configuration.
- Primary budget: 15,000 evaluations unless the selected suite specifies another official budget.
- Primary output: final error from the known optimum.
- Secondary outputs: convergence AUC, success rate, runtime, robustness, state/action behavior, and diversity.

Optional scalability study for the final two algorithms only:

- dimensions 10, 30, and 50;
- budget scaled with dimension;
- do not use the scalability study to select the algorithm after seeing its results.

## C3. Staged testing to control cost

### Gate 1 — Unit and smoke tests

- All correctness tests pass.
- Run 5 seeds on Sphere and Rastrigin with a small budget.
- Confirm deterministic reproduction, correct budget use, and monotone best-so-far history.

### Gate 2 — Five-function diagnostic screen

Use:

- Sphere;
- Rosenbrock;
- Rastrigin;
- Schwefel 2.26;
- noisy Quartic or another explicitly controlled noisy function.

Run 10 seeds. Reject changes that cause catastrophic regression or runtime failure.

### Gate 3 — Ten-function balanced screen

Use 10 functions spanning the categories in the final suite. Run 10 paired seeds.

Promote a change only if:

- it improves overall rank or convergence efficiency;
- its improvement is not driven by one function;
- it introduces no repeated numerical failures;
- any regressions are limited and understood.

### Gate 4 — Full 40-function confirmation

Run 10 paired seeds for the complete routine comparison. After selecting the strongest methods, optionally run 30 paired seeds only for the corrected baseline and final one or two methods:

- corrected baseline;
- best single learning change;
- best single movement change;
- best three-state classifier;
- best elite selector;
- at most two predeclared combinations.

This limits the final experiment to approximately 6–7 algorithms rather than benchmarking every unsuccessful idea 1,200 times.

## C4. Required performance metrics

For each function and algorithm report:

- best, median, mean, standard deviation, IQR, and worst final error;
- success rate at a predeclared tolerance;
- median evaluations to target when the target is reached;
- convergence curve against evaluations;
- area under the log-error convergence curve;
- runtime, but separately from objective-evaluation efficiency;
- failure and non-finite result count.

Use a safe log-error value:

```text
log_error = log10(max(abs(best_f - known_optimum), 1e-16))
```

For noisy benchmarks, define success using an estimated expected objective or a fixed resampling rule rather than a single noisy observation.

## C5. Statistical analysis

Use paired seeds.

Per function:

- paired Wilcoxon signed-rank test;
- Holm correction across the 40 functions;
- paired rank-biserial correlation or another paired effect size;
- win/tie/loss based on both statistical and practical significance.

Across functions:

- Friedman test;
- average-rank table;
- post-hoc comparison against the corrected baseline;
- critical-difference diagram if appropriate.

Do not conclude that an algorithm is better from mean values alone.

## C6. Predeclared interpretation rules

Classify each function result as:

- **Win:** corrected p-value below 0.05 and effect exceeds the predeclared practical threshold.
- **Loss:** corrected p-value below 0.05 in the opposite direction and effect exceeds the threshold.
- **Tie:** all other outcomes.

The final optimized TIEO should be considered successful only if it:

1. improves average rank across the 40 functions;
2. has more wins than losses against the corrected baseline;
3. improves convergence AUC or evaluations-to-target, not only terminal mean;
4. does not create a small set of catastrophic regressions;
5. remains reproducible and within the identical evaluation budget.

---

# Part D — Recommended experiment order

| Order | Configuration | Difference from preceding reference |
|---:|---|---|
| 1 | B0 Legacy baseline | Historical comparison only |
| 2 | B1 Corrected baseline | Correct benchmarks, RNG, acceptance, and FE accounting |
| 3 | B2 Efficient baseline | M01–M07 behavior-preserving changes |
| 4 | L1 | Magnitude-aware reward only, M08 |
| 5 | L2 | Outcome-based war reward only, M09 |
| 6 | L3 | Best learning-rate/gamma choice, M10–M12 |
| 7 | V1 | Reflective boundary handling only, if not already adopted |
| 8 | V2 | Decreasing autonomy weight only, M13 |
| 9 | V3 | Improved diplomacy interpolation only, M15 |
| 10 | S1 | Rank/quantile three states only, M18 |
| 11 | S2 | Robust z-score three states only, M19 |
| 12 | S3 | Best three-state method plus hysteresis, M20 |
| 13 | E1 | Inverse-rank target selection only, M22 |
| 14 | E2 | Softmax-rank target selection only, M23 |
| 15 | C1 | Best learning change + best movement change |
| 16 | C2 | C1 + best state classifier |
| 17 | C3 | C2 + best elite target selector |
| 18 | C4 | Best final combination + controlled restart, if needed |

At every step, compare against B1 using the same seeds and budget. Do not automatically carry every previous change forward; a change must earn inclusion through the staged gates.

---

# Part E — Expected outcomes versus measured outcomes

These are hypotheses, not performance claims:

| Change | Reasonable hypothesis |
|---|---|
| Restore greedy acceptance | Large recovery for the current three-state variants |
| Fair FE budget | May change rankings because war is much cheaper per action |
| Quantile/rank states | Most promising three-state candidate from current evidence |
| Robust z-score | May help when one tribe is an outlier |
| Relative deviation | Likely fragile near zero or with negative objectives unless redesigned |
| Soft elite targeting | May improve multimodal search without the premature convergence of hard elite copying |
| Magnitude-aware reward | May distinguish genuinely useful actions from tiny accidental improvements |
| Decreasing `w` | Likely to improve late exploitation while retaining early exploration |
| Reflective bounds | Likely to help functions whose good search directions interact with boundaries |
| Partial restart | Likely to help multimodal and composition functions, but can hurt easy unimodal convergence if triggered too early |

Actual performance on the 40 functions must be filled in only after the full confirmation experiment. The final report should include a 40-row win/tie/loss table, average ranks, critical-difference analysis, and convergence plots against objective evaluations.

## Immediate next implementation milestone

The first milestone should contain only:

1. canonical benchmark definitions;
2. corrected baseline TIEO;
3. exact evaluation counting and hard budget;
4. one seeded RNG;
5. automated tests;
6. one clean rank/quantile three-state classifier;
7. a runner that writes one structured result row per seed.

Do not implement elite ranking or combined mechanisms until this milestone reproduces stable baseline results and completes the five-function diagnostic screen.
