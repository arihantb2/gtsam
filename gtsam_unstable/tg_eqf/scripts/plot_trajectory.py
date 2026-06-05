#!/usr/bin/env python3
"""Plot TG-EqF trajectory example CSV: GT vs estimate and error time series."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory_csv(path: Path) -> dict[str, np.ndarray]:
    """Load trajectory CSV written by TG-EqF example executables."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    return {
        "t": col("t"),
        "gt_p": np.column_stack([col("gt_px"), col("gt_py"), col("gt_pz")]),
        "est_p": np.column_stack([col("est_px"), col("est_py"), col("est_pz")]),
        "gt_v": np.column_stack([col("gt_vx"), col("gt_vy"), col("gt_vz")]),
        "est_v": np.column_stack([col("est_vx"), col("est_vy"), col("est_vz")]),
    }


def rms(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))


def plot_trajectory(data: dict[str, np.ndarray], title: str, output_dir: Path) -> None:
    t = data["t"]
    pos_err = data["gt_p"] - data["est_p"]
    vel_err = data["gt_v"] - data["est_v"]
    pos_norm = np.linalg.norm(pos_err, axis=1)
    vel_norm = np.linalg.norm(vel_err, axis=1)

    stem = Path(title).stem

    # 1. XY trajectory comparison
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(data["gt_p"][:, 0], data["gt_p"][:, 1], "g-", label="Ground truth", lw=2)
    ax.plot(
        data["est_p"][:, 0],
        data["est_p"][:, 1],
        "b--",
        label="TG-EqF estimate",
        lw=2,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory comparison: {title}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_trajectory_xy.png", dpi=150)
    plt.close(fig)

    # 2. Position error vs time
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, pos_norm, "k-", label="||Δp||", lw=2)
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title(f"Position error: {title}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, np.abs(pos_err[:, 0]), label="|Δpx|")
    axes[1].plot(t, np.abs(pos_err[:, 1]), label="|Δpy|")
    axes[1].plot(t, np.abs(pos_err[:, 2]), label="|Δpz|")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Per-axis error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_position_error.png", dpi=150)
    plt.close(fig)

    # 3. Velocity error vs time
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, vel_norm, "k-", label="||Δv||", lw=2)
    axes[0].set_ylabel("Velocity error (m/s)")
    axes[0].set_title(f"Velocity error: {title}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, np.abs(vel_err[:, 0]), label="|Δvx|")
    axes[1].plot(t, np.abs(vel_err[:, 1]), label="|Δvy|")
    axes[1].plot(t, np.abs(vel_err[:, 2]), label="|Δvz|")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Per-axis error (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_velocity_error.png", dpi=150)
    plt.close(fig)

    print(f"RMS position error (m): {rms(pos_err):.6f}")
    print(f"RMS velocity error (m/s): {rms(vel_err):.6f}")
    print(f"Final position error (m): {pos_norm[-1]:.6f}")
    print(f"Final velocity error (m/s): {vel_norm[-1]:.6f}")
    print(f"Plots saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TG-EqF trajectory CSV (ground truth vs estimate)."
    )
    parser.add_argument("csv", type=Path, help="Path to trajectory CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG output (default: same as CSV)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    output_dir = args.output_dir if args.output_dir is not None else args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_trajectory_csv(args.csv)
    plot_trajectory(data, args.csv.name, output_dir)


if __name__ == "__main__":
    main()
