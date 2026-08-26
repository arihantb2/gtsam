#!/usr/bin/env python3
"""Extract SamoaWaypoints.h from a real vehicle_pose_est.csv trajectory.

Reads a pose CSV (columns: timestamp,tx,ty,tz,qw,qx,qy,qz), keeps the first
--duration seconds, smooths tx/ty/tz with a short moving average to remove
per-sample estimator jitter, resamples the smoothed path to --waypoints
points evenly spaced by cumulative arc length, moves the path into the
inertial frame (first waypoint at the origin, initial heading along nav +x,
gravity axis left alone), and writes the result as a
gtsam_unstable/examples_common/SamoaWaypoints.h header in the same style as
IMUScenarios.h's defaultSplineWaypoints().

Usage:
    extract_survey_waypoints.py INPUT.csv [--duration 300] [--waypoints 1500]
        [--smooth-window 2.0] [--output SamoaWaypoints.h]
"""
import argparse
import csv
import math
import os


def read_positions(path, duration):
    """Return (timestamps, positions) for rows within `duration` s of row 0."""
    timestamps, positions = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        t0 = None
        for row in reader:
            t = float(row["timestamp"])
            if t0 is None:
                t0 = t
            rel_t = t - t0
            if rel_t > duration:
                break
            timestamps.append(rel_t)
            positions.append(
                (float(row["tx"]), float(row["ty"]), float(row["tz"]))
            )
    return timestamps, positions


def initial_heading(positions, baseline_m=1.0):
    """Heading of the initial direction of travel, in the nav xy plane.

    Measured over a short arc-length baseline rather than off the first
    segment, so per-sample jitter cannot swing it. Deliberately the direction
    of TRAVEL, not the vehicle's logged attitude: SplineScenario holds
    attitude at identity, so the logged attitude is discarded anyway, and on
    this dive the two disagree by ~137 deg at t=0 (the AUV was not moving
    along its own x axis).
    """
    x0, y0 = positions[0][0], positions[0][1]
    for x, y, _ in positions[1:]:
        if math.hypot(x - x0, y - y0) >= baseline_m:
            return math.atan2(y - y0, x - x0)
    return math.atan2(positions[-1][1] - y0, positions[-1][0] - x0)


def to_inertial_frame(positions, heading0):
    """Recenter at the origin and align the initial heading with nav +x.

    Applies the rigid transform p -> Rz(-heading0) (p - p0), so the first
    waypoint lands exactly on the origin and the initial direction of travel
    points along the nav x axis -- matching SplineScenario's identity
    attitude, where body and nav axes coincide.

    Only the heading is removed, never roll/pitch: +z must stay the gravity
    (depth) axis, and a full 3-D de-rotation would tilt it, breaking both the
    gravity alignment the filters assume and the depth sensor's meaning. The
    vehicle's roll/pitch here are milliradian-scale anyway.
    """
    cos_y, sin_y = math.cos(-heading0), math.sin(-heading0)
    x0, y0, z0 = positions[0]
    out = []
    for x, y, z in positions:
        dx, dy, dz = x - x0, y - y0, z - z0
        out.append((cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy, dz))
    return out


def smooth(timestamps, positions, window_s):
    """Centered moving average over a `window_s`-second window, by sample count
    inferred from the local sample rate (irregular timestamps are fine)."""
    n = len(positions)
    smoothed = []
    half = window_s / 2.0
    lo = 0
    for i in range(n):
        while timestamps[i] - timestamps[lo] > half:
            lo += 1
        hi = i
        while hi + 1 < n and timestamps[hi + 1] - timestamps[i] <= half:
            hi += 1
        xs = [positions[k][0] for k in range(lo, hi + 1)]
        ys = [positions[k][1] for k in range(lo, hi + 1)]
        zs = [positions[k][2] for k in range(lo, hi + 1)]
        m = len(xs)
        smoothed.append((sum(xs) / m, sum(ys) / m, sum(zs) / m))
    return smoothed


def resample_by_arc_length(positions, num_waypoints):
    """Evenly-spaced-by-arc-length resample of a polyline to num_waypoints pts."""

    def dist(a, b):
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))

    cum = [0.0]
    for i in range(1, len(positions)):
        cum.append(cum[-1] + dist(positions[i - 1], positions[i]))
    total = cum[-1]
    if total == 0.0:
        return [positions[0]] * num_waypoints

    targets = [total * k / (num_waypoints - 1) for k in range(num_waypoints)]
    out = []
    j = 0
    for s in targets:
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        if j + 1 >= len(cum):
            out.append(positions[-1])
            continue
        seg = cum[j + 1] - cum[j]
        frac = 0.0 if seg == 0.0 else (s - cum[j]) / seg
        a, b = positions[j], positions[j + 1]
        out.append(tuple(a[k] + frac * (b[k] - a[k]) for k in range(3)))
    return out


def render_header(waypoints, source_path, duration, heading0):
    lines = []
    lines.append("/**")
    lines.append(" * @file SamoaWaypoints.h")
    lines.append(" * @brief Waypoints for the SamoaSurvey scenario, generated from a real")
    lines.append(" *        AUV trajectory. Regenerate with")
    lines.append(" *        scripts/extract_survey_waypoints.py; do not hand-edit.")
    lines.append(" *")
    lines.append(f" * Source: {source_path}")
    lines.append(f" * First {duration:g} s, smoothed and resampled to {len(waypoints)}")
    lines.append(" * waypoints evenly spaced by arc length, then rigidly moved into the")
    lines.append(" * inertial frame: translated so the first waypoint is exactly the")
    lines.append(f" * origin, and rotated about the gravity axis by {-heading0:+.4f} rad so")
    lines.append(" * the initial direction of travel points along nav +x. Only the heading")
    lines.append(" * is removed, so +z stays the gravity (depth) axis. Z-down NED, matching")
    lines.append(" * this repo's convention -- no sign conversion applied. Regenerate with:")
    lines.append(" *   python3 scripts/extract_survey_waypoints.py \\")
    lines.append(f" *       {source_path} --output SamoaWaypoints.h")
    lines.append(" */")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <gtsam/geometry/Point3.h>")
    lines.append("")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace imu_scenarios {")
    lines.append("")
    lines.append(
        f"/// First {duration:g} s of a real AUV survey trajectory ({os.path.basename(source_path)}),"
    )
    lines.append(
        "/// smoothed, resampled to evenly spaced points by arc length, and moved"
    )
    lines.append(
        "/// into the inertial frame (starts at the origin, initial heading along"
    )
    lines.append(
        "/// nav +x, gravity axis untouched). See file header for provenance."
    )
    lines.append("inline std::vector<gtsam::Vector3> samoaSurveyWaypoints() {")
    lines.append("  return {")
    per_line = 2
    for i in range(0, len(waypoints), per_line):
        chunk = waypoints[i : i + per_line]
        # round-then-add-0.0 normalizes IEEE negative zero, so the recentered
        # first waypoint prints as 0.000 rather than -0.000.
        entries = ", ".join(
            "gtsam::Vector3({:.3f}, {:.3f}, {:.3f})".format(
                *(round(v, 3) + 0.0 for v in pt)
            )
            for pt in chunk
        )
        lines.append(f"      {entries},")
    lines.append("  };")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace imu_scenarios")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to vehicle_pose_est.csv")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="seconds of trajectory to keep from the start")
    parser.add_argument("--waypoints", type=int, default=1500,
                        help="number of output waypoints (default 1500 = 5 Hz "
                             "over the default 300 s)")
    parser.add_argument("--smooth-window", type=float, default=2.0,
                        help="moving-average window in seconds")
    parser.add_argument("--output",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                            "SamoaWaypoints.h"),
                        help="output header path")
    args = parser.parse_args()

    timestamps, positions = read_positions(args.input, args.duration)
    if len(positions) < 2:
        raise SystemExit("not enough rows in the requested duration")

    smoothed = smooth(timestamps, positions, args.smooth_window)
    waypoints = resample_by_arc_length(smoothed, args.waypoints)
    # Recenter/align last, so the first waypoint is exactly the origin: the
    # moving average shifts the leading sample, so recentering before
    # smoothing would leave it slightly off.
    heading0 = initial_heading(waypoints)
    waypoints = to_inertial_frame(waypoints, heading0)

    header = render_header(waypoints, os.path.abspath(args.input),
                           args.duration, heading0)
    with open(args.output, "w") as f:
        f.write(header)
    print(f"wrote {len(waypoints)} waypoints to {args.output}")


if __name__ == "__main__":
    main()
