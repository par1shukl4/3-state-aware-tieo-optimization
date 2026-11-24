# Burgers PINN with Quantile + Local TIEO

This application uses the optimized Quantile + Local TIEO method as an outer
loop for continuous PINN hyperparameter optimization. Adam trains the neural
network weights; TIEO searches learning rate, weight decay, three physics-loss
weights and the `tanh` activation scale. The final 5% of outer evaluations are
reserved for bounded Powell refinement.

The solved problem is:

```text
u_t + u u_x - nu u_xx = 0,  nu = 0.01/pi
x in [-1, 1], t in [0, 1]
u(x, 0) = -sin(pi x)
u(-1, t) = u(1, t) = 0
```

This is the standard Burgers benchmark used in the original PINN work by
Raissi, Perdikaris and Karniadakis:
<https://doi.org/10.1016/j.jcp.2018.10.045>.

## Installation

From `TIEO_Master`:

```powershell
python -m pip install -r PINN_Burgers\requirements.txt
```

PyTorch can be installed separately using the command recommended for the
computer's CPU/CUDA configuration at <https://pytorch.org/get-started/locally/>.

## Verify the installation

```powershell
python -m PINN_Burgers.optimize_quantile_local --quick
```

The quick mode verifies the complete pipeline but is not a scientific run.

## Initial scientific run

```powershell
python -m PINN_Burgers.optimize_quantile_local
```

For a CUDA GPU:

```powershell
python -m PINN_Burgers.optimize_quantile_local --device cuda
```

The default search performs 60 complete PINN-training trials. Each candidate
receives the same collocation points, initialization seed, network structure
and number of Adam steps, making the outer objective deterministic. The best
hyperparameters are then retrained using an independent initialization and a
longer Adam + L-BFGS schedule.

## Outputs

Every run is preserved in the next `results_1`, `results_2`, ... folder and
includes:

- numerical reference and final PINN solution arrays;
- all hyperparameter trials and TIEO convergence CSVs;
- best hyperparameters, final metrics and model state;
- optimization convergence plot;
- hyperparameter search trace;
- reference, prediction and absolute-error heatmaps;
- solution cross-sections at five times;
- final Adam/L-BFGS loss curves.

The scientific result should eventually be repeated over several outer seeds.
The initial implementation intentionally includes only Quantile + Local TIEO,
as requested.
