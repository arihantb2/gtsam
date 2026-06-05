# TG-EqF trajectory example plotting

Simulation parameters, scenario definitions, and filter settings are documented
in [`../examples/README.md`](../examples/README.md).

## Workflow

```bash
# 1. Build and run a C++ example (from gtsam/build)
cmake --build . --target TGEqFStraightLineExample TGEqFCircleExample
./gtsam_unstable/tg_eqf/examples/TGEqFStraightLineExample --output tg_eqf_straight_line.csv
./gtsam_unstable/tg_eqf/examples/TGEqFCircleExample --output tg_eqf_circle.csv

# 2. Plot trajectories and errors
python ../gtsam_unstable/tg_eqf/scripts/plot_trajectory.py tg_eqf_straight_line.csv
python ../gtsam_unstable/tg_eqf/scripts/plot_trajectory.py tg_eqf_circle.csv --output-dir plots/
```

## Dependencies

- Python 3
- `numpy` (required for both scripts)
- `matplotlib` (plots only; `monte_carlo.py --no-plots` runs without it)

## Output figures

For each CSV, three PNG files are written:

- `*_trajectory_xy.png` — ground truth vs estimate in the XY plane
- `*_position_error.png` — position error norm and per-axis errors vs time
- `*_velocity_error.png` — velocity error norm and per-axis errors vs time

## Monte-Carlo analysis

`monte_carlo.py` sweeps the IMU-noise RNG seed over many runs of a scenario
executable and aggregates per-state error envelopes and filter-consistency
(per-group NEES) across the runs — for the navigation states (attitude,
position, velocity) and the bias states (gyro, accel, virtual-velocity).

```bash
python3 ../gtsam_unstable/tg_eqf/scripts/monte_carlo.py \
    --bin-dir gtsam_unstable/tg_eqf/examples \
    --scenarios circle straight_line --num-runs 50 \
    --gyro-noise 1e-3 --accel-noise 1e-2 \
    --gyro-bias-rw 1e-4 --accel-bias-rw 1e-3 --init-sigma 1e-3 \
    --output-dir mc/
```

The runs are IMU-only (no aiding), so bias is unobservable: the analysis shows
how the error grows and whether the filter covariance stays consistent with that
growth. **For NEES to be meaningful, drive the runs with stochastic noise**
(white noise and/or bias random walk) and a small `--init-sigma`; the example
wires those into the filter `Qc`/`Sigma0` so the covariance is physically
matched. A constant `--gyro-bias`/`--accel-bias` is a deterministic offset — use
the error-envelope view for it, not NEES.

Per scenario it writes (prefixed with the scenario stem in `--output-dir`):

- `*_mc_config.json` — full run metadata (params, seed list, executable + git
  commit, timestamp, per-run command template). Replay a run with `--config`.
- `*_mc_summary.csv` — per-group final/RMS error and mean ANEES with a
  consistency flag (ANEES ≈ 3 = χ²₃; below = conservative, above = optimistic).
- `*_mc_nav_errors.png`, `*_mc_bias_errors.png` — MC mean ±1σ error envelopes
  and RMSE per state group.
- `*_mc_consistency.png` — ANEES(t) per group vs the expected value and 95% band.

Key options: `--num-runs`, `--seed-start` (sweep range), `--config` (load
parameters from a prior `*_mc_config.json`; explicit flags override),
`--log-decim` (log every Nth step; default 10, to bound file sizes),
`--keep-csv` (retain per-run CSVs), `--no-plots` (emit only the summary + config,
for environments without a working matplotlib).
