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
- `numpy`
- `matplotlib`

## Output figures

For each CSV, three PNG files are written:

- `*_trajectory_xy.png` — ground truth vs estimate in the XY plane
- `*_position_error.png` — position error norm and per-axis errors vs time
- `*_velocity_error.png` — velocity error norm and per-axis errors vs time
