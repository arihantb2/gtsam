#!/usr/bin/env python3
"""Monte-Carlo analysis of TG-EqF navigation and bias states.

Sweeps the IMU-noise RNG seed over N runs of a scenario executable, then
aggregates per-state error envelopes and filter-consistency (per-group NEES)
across the runs. Navigation states are attitude / position / velocity; bias
states are gyro / accel / virtual-velocity.

The runs are IMU-only (no measurement updates), so bias is unobservable: the
analysis shows how the error grows and whether the filter covariance stays
consistent with that growth (NEES). For NEES to be meaningful, run the
scenarios with matched noise (e.g. --gyro-noise / --accel-noise /
--gyro-bias-rw / --accel-bias-rw and a small --init-sigma); the example wires
those into the filter Qc/Sigma0.

Plots need matplotlib; pass --no-plots to emit only the summary CSV + config
(useful where matplotlib is unavailable). numpy is always required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

# Scenario name -> (executable file name, output stem).
SCENARIOS = {
    "circle": ("TGEqFCircleExample", "tg_eqf_circle"),
    "straight_line": ("TGEqFStraightLineExample", "tg_eqf_straight_line"),
}

# Display label and physical unit per state group. Order matches the CSV's raw
# tangent order [att, vel, pos, bg, ba, bv]; eps/P columns are selected by name
# (eps_and_cov), so this list only sets display order.
GROUPS = ["att", "vel", "pos", "bg", "ba", "bv"]
GROUP_LABEL = {
    "att": ("Attitude", "rad"),
    "pos": ("Position", "m"),
    "vel": ("Velocity", "m/s"),
    "bg": ("Gyro bias", "rad/s"),
    "ba": ("Accel bias", "m/s^2"),
    "bv": ("Virtual-vel bias", "m/s"),
}

# Sim parameters forwarded verbatim to the executable (flag -> argparse dest).
SCALAR_PARAMS = [
    ("--duration", "duration"),
    ("--dt", "dt"),
    ("--gyro-noise", "gyro_noise"),
    ("--accel-noise", "accel_noise"),
    ("--gyro-bias-rw", "gyro_bias_rw"),
    ("--accel-bias-rw", "accel_bias_rw"),
    ("--init-sigma", "init_sigma"),
    ("--log-decim", "log_decim"),
]
VECTOR_PARAMS = [("--gyro-bias", "gyro_bias"), ("--accel-bias", "accel_bias")]


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a TG-EqF CSV into a {column_name: 1-D array} dict."""
    with path.open(newline="") as f:
        names = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {name: arr[:, i] for i, name in enumerate(names)}


def stack3(d: dict[str, np.ndarray], cols: list[str]) -> np.ndarray:
    """Stack three named columns into an [T, 3] array."""
    return np.column_stack([d[c] for c in cols])


def physical_errors(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-group physical error [T, 3] = estimate - truth."""
    est_p = stack3(d, ["est_px", "est_py", "est_pz"])
    gt_p = stack3(d, ["gt_px", "gt_py", "gt_pz"])
    est_v = stack3(d, ["est_vx", "est_vy", "est_vz"])
    gt_v = stack3(d, ["gt_vx", "gt_vy", "gt_vz"])
    att = stack3(d, ["att_err_x", "att_err_y", "att_err_z"])
    bg = stack3(d, ["est_bgx", "est_bgy", "est_bgz"]) - stack3(
        d, ["true_bgx", "true_bgy", "true_bgz"]
    )
    ba = stack3(d, ["est_bax", "est_bay", "est_baz"]) - stack3(
        d, ["true_bax", "true_bay", "true_baz"]
    )
    bv = stack3(d, ["est_bvx", "est_bvy", "est_bvz"])  # virtual: truth is 0
    return {"att": att, "pos": est_p - gt_p, "vel": est_v - gt_v,
            "bg": bg, "ba": ba, "bv": bv}


def eps_and_cov(d: dict[str, np.ndarray], g: str) -> tuple[np.ndarray, np.ndarray]:
    """Origin-frame error eps [T, 3] and covariance block P [T, 3, 3]."""
    eps = stack3(d, [f"eps_{g}_x", f"eps_{g}_y", f"eps_{g}_z"])
    a, b, c, e, f, h = (d[f"P_{g}_{ij}"] for ij in
                        ("00", "01", "02", "11", "12", "22"))
    T = len(a)
    P = np.empty((T, 3, 3))
    P[:, 0, 0] = a
    P[:, 0, 1] = P[:, 1, 0] = b
    P[:, 0, 2] = P[:, 2, 0] = c
    P[:, 1, 1] = e
    P[:, 1, 2] = P[:, 2, 1] = f
    P[:, 2, 2] = h
    return eps, P


# --------------------------------------------------------------------------- #
# Running the executable
# --------------------------------------------------------------------------- #
def build_command(exe: Path, args, seed: int, out: Path) -> list[str]:
    cmd = [str(exe), "--seed", str(seed), "--output", str(out)]
    for flag, dest in SCALAR_PARAMS:
        cmd += [flag, str(getattr(args, dest))]
    for flag, dest in VECTOR_PARAMS:
        v = getattr(args, dest)
        cmd += [flag, ",".join(str(x) for x in v)]
    return cmd


def run_scenario(name: str, args, tmp: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Run N seeds; return (time [T], data [N, T, C], seeds, column_names)."""
    exe_name, _ = SCENARIOS[name]
    exe = Path(args.bin_dir) / exe_name
    if not exe.is_file():
        raise SystemExit(f"Executable not found: {exe}")

    seeds = [args.seed_start + i for i in range(args.num_runs)]
    runs, names = [], None
    for i, seed in enumerate(seeds):
        out = tmp / f"{name}_seed{seed}.csv"
        cmd = build_command(exe, args, seed, out)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"Run failed (seed {seed}):\n{proc.stderr}")
        d = load_csv(out)
        if names is None:
            names = list(d.keys())
            t0 = d["t"]
        elif d["t"].shape != t0.shape:
            raise SystemExit(f"Run {seed} has a different time base")
        runs.append(np.column_stack([d[n] for n in names]))
        print(f"  [{i + 1}/{len(seeds)}] seed {seed} ok", flush=True)
    return t0, np.stack(runs), seeds, names


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def chi2_interval(dof: int, p: float = 0.95) -> tuple[float, float]:
    """Two-sided chi^2 interval via the Wilson-Hilferty approximation."""
    lo, hi = (1 - p) / 2, 1 - (1 - p) / 2
    z_lo, z_hi = NormalDist().inv_cdf(lo), NormalDist().inv_cdf(hi)
    c = 2.0 / (9.0 * dof)
    return (dof * (1 - c + z_lo * c**0.5) ** 3,
            dof * (1 - c + z_hi * c**0.5) ** 3)


def analyze(t: np.ndarray, data: np.ndarray, names: list[str]) -> dict:
    """Compute per-group error envelopes, RMSE, and ANEES(t) over runs."""
    dicts = [{n: data[r, :, i] for i, n in enumerate(names)}
             for r in range(data.shape[0])]
    N = len(dicts)
    out = {"t": t, "N": N, "groups": {}}
    for g in GROUPS:
        err = np.stack([physical_errors(d)[g] for d in dicts])  # [N,T,3]
        norm = np.linalg.norm(err, axis=2)                       # [N,T]
        nees = np.empty((N, len(t)))
        for r, d in enumerate(dicts):
            eps, P = eps_and_cov(d, g)
            nees[r] = np.einsum("ti,tij,tj->t", eps, np.linalg.inv(P), eps)
        out["groups"][g] = {
            "err_mean": norm.mean(axis=0),
            "err_std": norm.std(axis=0),
            "rmse": np.sqrt((norm**2).mean(axis=0)),
            "anees": nees.mean(axis=0),
        }
    lo, hi = chi2_interval(3 * N)
    out["anees_bounds"] = (lo / N, hi / N)  # per-run ANEES bounds (expect ~3)
    return out


# --------------------------------------------------------------------------- #
# Output: summary, config, plots
# --------------------------------------------------------------------------- #
def write_summary(res: dict, path: Path, config_name: str) -> None:
    lo, hi = res["anees_bounds"]
    with path.open("w", newline="") as f:
        f.write(f"# TG-EqF Monte-Carlo summary; config: {config_name}\n")
        f.write(f"# runs={res['N']}; ANEES expected 3.0, 95% in "
                f"[{lo:.2f},{hi:.2f}]\n")
        f.write("group,unit,final_err_mean,final_err_std,time_avg_rmse,"
                "final_anees,mean_anees,consistent\n")
        for g in GROUPS:
            gd = res["groups"][g]
            label, unit = GROUP_LABEL[g]
            mean_anees = float(gd["anees"].mean())
            ok = lo <= mean_anees <= hi
            f.write(f"{label},{unit},{gd['err_mean'][-1]:.6g},"
                    f"{gd['err_std'][-1]:.6g},{gd['rmse'].mean():.6g},"
                    f"{gd['anees'][-1]:.4g},{mean_anees:.4g},{int(ok)}\n")


def write_config(cfg: dict, path: Path) -> None:
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def make_config(name: str, args, seeds: list[int], exe: Path) -> dict:
    commit = "unknown"
    try:
        commit = subprocess.run(
            ["git", "-C", str(exe.resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        pass
    return {
        "scenario": name,
        "executable": str(exe),
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_runs": args.num_runs,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "params": {dest: getattr(args, dest)
                   for _, dest in SCALAR_PARAMS + VECTOR_PARAMS},
        "command_template": build_command(exe, args, "<seed>",
                                          Path("<output>")),
    }


def make_plots(res: dict, stem: str, out_dir: Path, config_name: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [plots skipped: matplotlib unavailable: {exc}]")
        return False
    t = res["t"]

    def err_figure(groups: list[str], fname: str, title: str) -> None:
        fig, axes = plt.subplots(len(groups), 1, figsize=(9, 3 * len(groups)),
                                 sharex=True)
        for ax, g in zip(np.atleast_1d(axes), groups):
            gd = res["groups"][g]
            label, unit = GROUP_LABEL[g]
            ax.plot(t, gd["err_mean"], "b-", lw=2, label="MC mean ||error||")
            ax.fill_between(t, gd["err_mean"] - gd["err_std"],
                            gd["err_mean"] + gd["err_std"], color="b",
                            alpha=0.2, label="MC ±1σ")
            ax.plot(t, gd["rmse"], "r--", lw=1.5, label="RMSE")
            ax.set_ylabel(f"{label} ({unit})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("time (s)")
        np.atleast_1d(axes)[0].set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    err_figure(["att", "pos", "vel"], f"{stem}_mc_nav_errors.png",
               f"Navigation error (MC, {res['N']} runs): {stem}")
    err_figure(["bg", "ba", "bv"], f"{stem}_mc_bias_errors.png",
               f"Bias error (MC, {res['N']} runs): {stem}")

    lo, hi = res["anees_bounds"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for g in GROUPS:
        ax.plot(t, res["groups"][g]["anees"], lw=1.5, label=GROUP_LABEL[g][0])
    ax.axhline(3.0, color="k", ls="-", lw=1, label="expected (χ²₃)")
    ax.axhspan(lo, hi, color="k", alpha=0.1, label="95% band")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ANEES (per group, dof=3)")
    ax.set_title(f"Filter consistency (NEES): {stem}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_mc_consistency.png", dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-dir", required=True,
                   help="Directory holding the example executables")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                   choices=list(SCENARIOS), help="Scenarios to run")
    p.add_argument("--num-runs", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--config", type=Path, default=None,
                   help="JSON file of parameters; explicit flags override it")
    p.add_argument("--keep-csv", action="store_true",
                   help="Keep per-run CSVs (in the output dir) instead of temp")
    p.add_argument("--no-plots", action="store_true")
    # Sim passthrough (defaults mirror the C++ RunOptions defaults).
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--gyro-bias", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--accel-bias", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--gyro-noise", type=float, default=0.0)
    p.add_argument("--accel-noise", type=float, default=0.0)
    p.add_argument("--gyro-bias-rw", type=float, default=0.0)
    p.add_argument("--accel-bias-rw", type=float, default=0.0)
    p.add_argument("--init-sigma", type=float, default=0.1)
    p.add_argument("--log-decim", type=int, default=10)
    return p


def apply_config(args, parser) -> None:
    """Load --config JSON, but let explicitly-passed CLI flags win."""
    if args.config is None:
        return
    cfg = json.loads(Path(args.config).read_text())
    given = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:]
             if a.startswith("--")}
    flat = {**cfg.get("params", {}),
            **{k: cfg[k] for k in ("num_runs", "seed_start") if k in cfg}}
    for key, val in flat.items():
        if key not in given and hasattr(args, key):
            setattr(args, key, val)


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    apply_config(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.scenarios:
        _, stem = SCENARIOS[name]
        exe = Path(args.bin_dir) / SCENARIOS[name][0]
        print(f"=== {name}: {args.num_runs} runs ===")

        ctx = (args.output_dir if args.keep_csv else
               Path(tempfile.mkdtemp(prefix=f"mc_{name}_")))
        res = None
        try:
            t, data, seeds, names = run_scenario(name, args, ctx)
            res = analyze(t, data, names)
        finally:
            if not args.keep_csv:
                shutil.rmtree(ctx, ignore_errors=True)

        cfg = make_config(name, args, seeds, exe)
        cfg_path = args.output_dir / f"{stem}_mc_config.json"
        write_config(cfg, cfg_path)
        write_summary(res, args.output_dir / f"{stem}_mc_summary.csv",
                      cfg_path.name)
        if not args.no_plots:
            make_plots(res, stem, args.output_dir, cfg_path.name)

        lo, hi = res["anees_bounds"]
        print(f"  config:  {cfg_path}")
        print(f"  summary: {args.output_dir / (stem + '_mc_summary.csv')}")
        print(f"  ANEES 95% band [{lo:.2f}, {hi:.2f}] (expected 3.0):")
        for g in GROUPS:
            ma = float(res["groups"][g]["anees"].mean())
            flag = "ok" if lo <= ma <= hi else ("OPT" if ma > hi else "cons")
            print(f"    {GROUP_LABEL[g][0]:<18} "
                  f"final_rmse={res['groups'][g]['rmse'][-1]:.4g} "
                  f"mean_ANEES={ma:.3g} [{flag}]")


if __name__ == "__main__":
    main()
