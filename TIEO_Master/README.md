# TIEO Master

This folder is the canonical home for the corrected baseline and all future TIEO variants.

## Files

- `tieo_core.py` — shared TIEO operators, tabular Q-learning, evaluation budget and logging.
- `benchmarks.py` — the fixed 40-function benchmark registry.
- `experiment_utils.py` — repeated runs, mean/std summaries, CSV/XLSX output and plots.
- `baseline_tieo.py` — corrected two-state baseline.
- `quantile_tieo.py` — baseline with quantile three-state classification only.
- `adaptive_zscore_tieo.py` — baseline with adaptive z-score classification only.
- `relative_deviation_tieo.py` — baseline with relative-deviation classification only.
- `evaluation_aware_reward_tieo.py` — baseline states with magnitude- and evaluation-aware Q rewards.
- `stagnation_restart_tieo.py` — baseline with a stagnation-triggered restart of the worst 5%.
- `decreasing_autonomy_weight_tieo.py` — baseline with autonomy weight decreasing from 0.9 to 0.15.
- `quantile_local_refinement_tieo.py` — quantile method with a final 5% bounded Powell refinement.
- `quantile_decreasing_autonomy_tieo.py` — quantile method with autonomy weight decreasing from 0.5 to 0.15.
- `quantile_adaptive_thresholds_tieo.py` — quantile thresholds moving from q20/q80 to q33/q67.
- `quantile_minimum_occupancy_tieo.py` — excludes tribes below five members from threshold estimation and classifies them with a two-state fallback.
- `quantile_diversity_restart_tieo.py` — quantile method with a stagnation- and diversity-triggered 10% restart.
- `compare_quantiles.py` — compares the original quantile method with exactly these five variants.
- `compare_baseline_local.py` — paired baseline versus quantile-local comparison with additional diagnostic graphs.
- `standard_optimizers.py` — budget-controlled SA, PSO, DE, ABC, WPS, SSA, SBO, MPA, SCA and GSA implementations.
- `compare_all_algorithms.py` — compares the ten reference algorithms with baseline TIEO and quantile plus local refinement.
- `PINN_Burgers/` — scientific-ML application using Quantile + Local TIEO to optimize Burgers-equation PINN hyperparameters and generate solution visualizations.
- `compare_all.py` — runs all registered methods and creates comparative results.
- `validate_setup.py` — fast benchmark, budget and reproducibility checks.

All methods use identical operators, greedy acceptance, random seeds, evaluation
budgets, target selection and benchmark definitions. Each experimental file enables
only its named change, so its effect can be measured independently.

## Installation

From this folder:

```powershell
python -m pip install -r requirements.txt
python validate_setup.py
```

## Standard one-run comparison

By default, every function is run once for every method:

```powershell
python compare_all.py
```

This runs all 40 benchmarks with all seven methods and a 15,000-evaluation
budget per method/function pair. It performs approximately 4.2 million
objective evaluations.

Every command-line execution is preserved automatically in the next available
folder: `results_1`, `results_2`, `results_3`, and so on. Existing numbered
folders are never overwritten. To choose an exact folder instead, pass
`--output-root my_results`; an explicitly selected folder retains the normal
overwrite/reuse behavior.

For a quick two-function smoke comparison:

```powershell
python compare_all.py --budget 500 --population 20 --benchmarks 1,19
```

## Compare only quantile methods

Run the original quantile algorithm and its five isolated variants on all 40
benchmarks with one run each:

```powershell
python compare_quantiles.py
```

For a quick selected-function comparison:

```powershell
python compare_quantiles.py --budget 500 --population 20 --benchmarks 11,15,17,26,29,31,38,40
```

The results are written to the next automatic `results_N` folder under
`quantile_comparison`.

## Five-seed baseline versus best quantile

Run only baseline TIEO and quantile with local refinement using five paired
seeds per benchmark:

```powershell
python compare_baseline_local.py --runs 5
```

In addition to the normal comparison figures and per-function convergence
plots, this produces a paired error scatter, per-function improvement bars,
category improvement boxplot and convergence-improvement overview.

## Twelve-algorithm comparison

Compare baseline TIEO, quantile plus local refinement, SA, PSO, DE, ABC, WPS,
SSA, SBO, MPA, SCA and GSA with five paired seeds:

```powershell
python compare_all_algorithms.py --runs 5 --sensitivity-runs 1
```

The output includes convergence curves for every selected function, detailed
and overall mean/std/runtime/rank/score tables, an overall algorithm-score
graph, TIEO fitness-improvement probabilities, and sensitivity graphs for
maximum tribal number at `w=0.5` and learning weight at `N=5`. Algorithm score
is a normalized benchmark-rank score from 0 to 100. Use `--skip-sensitivity`
when only the twelve-algorithm comparison is required.

## Optional repeated runs

Repeated runs are disabled by default. Request them explicitly when you want
standard deviations or statistical tests based on multiple random trials:

```powershell
python compare_all.py --runs 5
python compare_all.py --runs 30
```

With one run, mean, median, best and worst are the same result, and standard
deviation is recorded as zero.

## Run one method

```powershell
python baseline_tieo.py
python quantile_tieo.py
python adaptive_zscore_tieo.py
python relative_deviation_tieo.py
python evaluation_aware_reward_tieo.py
python stagnation_restart_tieo.py
python decreasing_autonomy_weight_tieo.py
```

The evaluation-aware reward method retains the baseline two-state classifier
and all baseline operators. Its only algorithmic change is:

```text
scale = max(abs(old_stat), population_MAD, 1e-12)
reward = clip((old_stat - new_stat) / scale, -1, 1) / evaluations_used
```

Autonomy and diplomacy compare accepted tribe fitness before and after the
operation. War uses the captured individual's actual before/after fitness, so
a worsening war move receives a negative reward.

The stagnation-restart method preserves the current best individual and
uniformly reinitializes the worst 5% after 1,500 evaluations without a
meaningful improvement, with at most two restarts. Restart evaluations count
toward the same fixed evaluation budget.

The decreasing-weight method uses a linear evaluation-progress schedule only
for the autonomy operator:

```text
w = 0.9 + (0.15 - 0.9) * evaluations_used / evaluation_budget
```

Useful options:

```text
--benchmarks 1,11,19,22,40   run selected benchmark IDs
--runs 5                      explicitly request five independent runs
--dimension 30               dimension of scalable functions
--population 50              population size
--max-tribes 5               maximum initial tribe count
--base-seed 1000             shared starting seed
--no-plots                    skip individual method plots
--quiet                       reduce console output
```

## Outputs

Each method folder contains:

- `all_runs.csv` — one row per benchmark and seed, including Q-tables and action statistics;
- `summary.csv` and `summary.xlsx` — mean, standard deviation, median, best, worst, IQR and errors;
- `convergence_runs.csv` and `convergence_summary.csv`;
- `mean_error_summary.png`;
- one convergence plot per benchmark.

The comparison folder contains:

- combined runs and summaries;
- paired Wilcoxon tests with Holm correction;
- benchmark rank table and average ranks;
- mean-error, average-rank and win/tie/loss graphs;
- comparative convergence plots for every function.

## Adding a future method

Add a new small entry file beside the existing method files. Implement its isolated mechanism in the shared core or as a configuration strategy, then register the method in `experiment_utils.METHODS`. Do not copy the complete optimizer into another notebook or change benchmark formulas inside a method file.
