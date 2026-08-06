/**
 * @file  IMUScenarios.h
 * @brief Filter-agnostic ground-truth trajectory factory shared by the
 *        tg_eqf, tfg_inekf, and mekf example runners.
 *
 * Each factory returns a concrete gtsam::Scenario describing one canonical
 * trajectory. Defining them in one place guarantees every filter is driven
 * through the IDENTICAL ground truth (same speeds, rates, accelerations), so a
 * cross-filter comparison reflects filter behaviour only -- not incidental
 * differences in how each example wired up its scenario.
 *
 * Process-noise PSDs for the runners live in ImuProcessNoise.h (same CLI → PSD
 * mapping for tg_eqf, tfg_inekf, and mekf).
 *
 * Usage:
 *   const auto scenario = imu_scenarios::circle();
 *   runScenario(scenario, opts, "Circle");
 *
 * The returned object is a concrete Scenario subclass; bind it to a named local
 * (so its lifetime spans the runScenario call) and pass by const reference.
 */
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/Scenario.h>

#include <unsupported/Eigen/Splines>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace imu_scenarios {

// --- Canonical trajectory parameters (shared across all filters) ------------
// Kept here, not in the individual example mains, so every filter sees the same
// numbers. Adjust once to retune every runner consistently.
inline constexpr double kConstSpeed =
    1.0;  // ConstantVelocity: cruise m/s, body x
inline constexpr double kPureYawRate = 0.5;  // PureRotation: rad/s, body z

inline constexpr double kCircleRadius = 10.0;  // Circle: m
inline constexpr double kCircleSpeed = 1.0;    // Circle: cruise m/s

inline constexpr double kTurnSpeed0 =
    0.0;  // CoordinatedTurn: initial m/s (starts at rest)
inline constexpr double kTurnYawRate = 0.3;  // CoordinatedTurn: rad/s, body z

inline constexpr double kLineAccel = 0.1;  // StraightLine: m/s^2
inline constexpr double kLineSpeed0 =
    0.0;  // StraightLine: initial m/s (starts at rest)

inline constexpr double kVertAccel = 0.2;   // Vertical: m/s^2 along nav z
inline constexpr double kVertSpeed0 = 0.0;  // Vertical: initial m/s

inline constexpr double kSinSpeed = 1.0;  // Sinusoid: cruise m/s along nav x
inline constexpr double kSinAmplitude = 2.0;  // Sinusoid: m lateral (nav y)
inline constexpr double kSinPeriod = 10.0;    // Sinusoid: s per lateral cycle

inline constexpr double kNavCruiseSpeed =
    2.0;  // TypicalNavigation: cruise m/s, body x
inline constexpr double kNavTurnRate =
    0.4;  // TypicalNavigation: yaw rate during a turn, rad/s

inline constexpr double kSplinePathDuration =
    40.0;  // WaypointSpline: warped-time seconds to traverse the waypoint list
           // (u:0->1). Sized > the default 30 s --duration so a default run
           // never reaches the final-waypoint clamp; see SplineScenario.

// Every scenario is required to start at rest (R=I, p=0, v=0) so filters can be
// initialized at an identity/zero state. AcceleratingScenario-based factories
// (CoordinatedTurn, StraightLine, Vertical) already ramp velocity linearly from
// v0, so setting their initial-speed constants to 0 above suffices. Scenarios
// with a constant (nonzero) cruise speed by construction -- the constant-twist
// ones (ConstantVelocity, Circle) and the closed-form sinusoid -- instead reach
// cruise speed via a smooth spin-up over kRampDuration; see evalRamp,
// RampedTwistScenario, and SinusoidScenario below.
inline constexpr double kRampDuration = 2.0;  // s, shared spin-up duration

/// Smooth spin-up profile shared by every constant-speed scenario. `S(t)` is an
/// arc-length reparameterization: `S(0)=0`, `S'(0)=S''(0)=0` (fully at rest),
/// and for `t >= T` it reduces to `S(t) = t - T/2` with `S'=1, S''=0` (exact
/// cruise, C2-continuous stitch at `t=T`). Built from the quintic smootherstep
/// `f(x) = 6x^5 - 15x^4 + 10x^3` (`S' = f(t/T)`, `S = integral of f`, `S'' =
/// f'/T`). https://iquilezles.org/articles/smoothsteps/ 
struct RampProfile {
  double S;
  double Sdot;
  double Sddot;
};

inline RampProfile evalRamp(double t, double T) {
  if (t <= 0.0) {
    return RampProfile{0.0, 0.0, 0.0};
  }
  if (t >= T) {
    return RampProfile{t - T / 2.0, 1.0, 0.0};
  }
  const double x = t / T;
  const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x, x6 = x5 * x;
  return RampProfile{T * (x6 - 3.0 * x5 + 2.5 * x4),
                     6.0 * x5 - 15.0 * x4 + 10.0 * x3,
                     (30.0 * x4 - 60.0 * x3 + 30.0 * x2) / T};
}

/// Constant body-frame twist [w, v_body], smoothly spun up from rest over
/// `evalRamp`'s duration `T` rather than applied from t=0 (unlike
/// gtsam::ConstantTwistScenario). Because twist is a single fixed se(3)
/// element, `pose(t) = nTb0 * Expmap(twist * S(t))` is an EXACT re-timing of
/// the same one-parameter subgroup (not an approximation), so cruise-phase (t
/// >= T) kinematics -- and thus radius/speed -- exactly match
/// ConstantTwistScenario.
class RampedTwistScenario : public gtsam::Scenario {
 public:
  RampedTwistScenario(const gtsam::Vector3& w, const gtsam::Vector3& v_body,
                      double rampDuration,
                      const gtsam::Pose3& nTb0 = gtsam::Pose3())
      : w_(w),
        v_body_(v_body),
        a_b_(w.cross(v_body)),
        T_(rampDuration),
        nTb0_(nTb0) {}

  gtsam::Pose3 pose(double t) const override {
    const gtsam::Vector6 twist = (gtsam::Vector6() << w_, v_body_).finished();
    return nTb0_ * gtsam::Pose3::Expmap(twist * evalRamp(t, T_).S);
  }
  gtsam::Vector3 omega_b(double t) const override {
    return w_ * evalRamp(t, T_).Sdot;
  }
  gtsam::Vector3 velocity_n(double t) const override {
    return rotation(t).matrix() * v_body_ * evalRamp(t, T_).Sdot;
  }
  gtsam::Vector3 acceleration_n(double t) const override {
    const RampProfile r = evalRamp(t, T_);
    return rotation(t) * (a_b_ * r.Sdot * r.Sdot + v_body_ * r.Sddot);
  }

 private:
  const gtsam::Vector3 w_, v_body_, a_b_;
  const double T_;
  const gtsam::Pose3 nTb0_;
};

/// Closed-form lateral-sinusoid scenario (identity attitude, no rotation),
/// smoothly spun up from rest by time-warping the whole argument through the
/// same `evalRamp` used by `RampedTwistScenario`. For `t >= kRampDuration` this
/// reduces exactly to the plain (un-ramped) sinusoid, shifted by `kRampDuration
/// / 2`:
///   p(t)    = ( v S,        A sin(w S),                              0 )
///   v_n(t)  = ( v Sdot,     A w cos(w S) Sdot,                        0 )
///   a_n(t)  = ( v Sddot,    A w (cos(w S) Sddot - w sin(w S) Sdot^2), 0 )
/// where S = S(t), Sdot = S'(t), Sddot = S''(t) from evalRamp(t,
/// kRampDuration).
class SinusoidScenario : public gtsam::Scenario {
 public:
  SinusoidScenario(double forward_speed, double amplitude, double omega)
      : v_(forward_speed), A_(amplitude), w_(omega) {}

  gtsam::Pose3 pose(double t) const override {
    const double S = evalRamp(t, kRampDuration).S;
    return gtsam::Pose3(gtsam::Rot3(),
                        gtsam::Point3(v_ * S, A_ * std::sin(w_ * S), 0.0));
  }
  gtsam::Vector3 omega_b(double /*t*/) const override {
    return gtsam::Vector3::Zero();
  }
  gtsam::Vector3 velocity_n(double t) const override {
    const RampProfile r = evalRamp(t, kRampDuration);
    return gtsam::Vector3(v_ * r.Sdot, A_ * w_ * std::cos(w_ * r.S) * r.Sdot,
                          0.0);
  }
  gtsam::Vector3 acceleration_n(double t) const override {
    const RampProfile r = evalRamp(t, kRampDuration);
    const double lateral = A_ * w_ *
                           (std::cos(w_ * r.S) * r.Sddot -
                            w_ * std::sin(w_ * r.S) * r.Sdot * r.Sdot);
    return gtsam::Vector3(v_ * r.Sddot, lateral, 0.0);
  }

 private:
  const double v_, A_, w_;
};

/// Waypoint trajectory: a cubic C2 spline interpolated through a list of
/// nav-frame position waypoints, driving position/velocity/acceleration by
/// ANALYTIC differentiation of that one spline (so the three channels are
/// exactly self-consistent, as ScenarioRunner's IMU synthesis requires -- unlike
/// gtsam::DiscreteScenario, which interpolates each channel independently). The
/// spline parameter u in [0,1] is mapped to physical time through the shared
/// `evalRamp` time-warp `S(t)` (u = S(t) / pathDuration), so the path spins up
/// smoothly from rest -- pose(0)=origin, v(0)=a(0)=0 -- matching the rest-start
/// invariant every other factory obeys. Attitude is held at identity (like
/// SinusoidScenario), so omega_b = 0 and the accelerometer sees the path's
/// nav-frame specific force; rotation is exercised by the circle / coordinated
/// turn / typical-navigation factories instead. Past the final waypoint
/// (u >= 1) the pose is clamped and held at rest; size pathDuration so a run
/// stays within the path (the default does).
class SplineScenario : public gtsam::Scenario {
 public:
  /// @param waypoints  >= 2 nav-frame positions; the path is auto-translated so
  ///                   pose(0) sits at the origin (rest-start invariant).
  /// @param pathDuration  warped-time seconds to traverse waypoints.front() ->
  ///                   waypoints.back() (spline parameter u: 0 -> 1). Must be
  ///                   > 0. Larger => slower traversal.
  /// @param rampDuration  rest-to-cruise spin-up (shared evalRamp profile).
  SplineScenario(const std::vector<gtsam::Vector3>& waypoints,
                 double pathDuration, double rampDuration = kRampDuration)
      : pathDuration_(pathDuration), rampDuration_(rampDuration) {
    const std::size_t n = waypoints.size();
    if (n < 2) {
      throw std::invalid_argument(
          "SplineScenario needs at least 2 waypoints.");
    }
    if (!(pathDuration_ > 0.0)) {
      throw std::invalid_argument("SplineScenario pathDuration must be > 0.");
    }
    Eigen::Array<double, 3, Eigen::Dynamic> pts(3, n);
    for (std::size_t i = 0; i < n; ++i) {
      pts.col(i) = waypoints[i].array();
    }
    // Cubic gives a C2 (continuous-acceleration) path; drop the degree only
    // when there are too few waypoints to support it.
    const Eigen::Index degree =
        std::min<Eigen::Index>(3, static_cast<Eigen::Index>(n) - 1);
    spline_ = Eigen::SplineFitting<Spline3>::Interpolate(pts, degree);
    origin_ = spline_(0.0).matrix();               // == waypoints.front()
    finalPos_ = spline_(1.0).matrix() - origin_;   // held past the last waypoint
  }

  gtsam::Pose3 pose(double t) const override {
    return gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(position_n(t)));
  }
  gtsam::Vector3 omega_b(double /*t*/) const override {
    return gtsam::Vector3::Zero();
  }
  gtsam::Vector3 velocity_n(double t) const override {
    const UParam u = mapTime(t);
    if (u.clamped) {
      return gtsam::Vector3::Zero();
    }
    return dpdu(u.u) * u.udot;
  }
  gtsam::Vector3 acceleration_n(double t) const override {
    const UParam u = mapTime(t);
    if (u.clamped) {
      return gtsam::Vector3::Zero();
    }
    const auto der = spline_.derivatives(u.u, 2);
    const gtsam::Vector3 d1 = der.col(1).matrix();
    const gtsam::Vector3 d2 = der.col(2).matrix();
    return d2 * (u.udot * u.udot) + d1 * u.uddot;
  }

 private:
  typedef Eigen::Spline<double, 3> Spline3;

  gtsam::Vector3 position_n(double t) const {
    const UParam u = mapTime(t);
    return u.clamped && u.u >= 1.0 ? finalPos_
                                   : (spline_(u.u).matrix() - origin_);
  }
  gtsam::Vector3 dpdu(double u) const {
    return spline_.derivatives(u, 1).col(1).matrix();
  }

  /// Spline parameter u = S(t)/pathDuration and its time derivatives, where S is
  /// the shared rest-start ramp. `clamped` marks t <= 0 (held at origin) or
  /// u >= 1 (held at the final waypoint) -- both rest states (v = a = 0).
  struct UParam {
    double u, udot, uddot;
    bool clamped;
  };
  UParam mapTime(double t) const {
    const RampProfile r = evalRamp(t, rampDuration_);
    const double u = r.S / pathDuration_;
    if (u <= 0.0) {
      return {0.0, 0.0, 0.0, true};
    }
    if (u >= 1.0) {
      return {1.0, 0.0, 0.0, true};
    }
    return {u, r.Sdot / pathDuration_, r.Sddot / pathDuration_, false};
  }

  const double pathDuration_, rampDuration_;
  Spline3 spline_;
  gtsam::Vector3 origin_ = gtsam::Vector3::Zero();
  gtsam::Vector3 finalPos_ = gtsam::Vector3::Zero();
};

/// One leg of a piecewise-constant-twist ground path: constant body angular
/// rate `w` and body-frame velocity `v_body`, held for `duration_s` seconds.
struct NavLeg {
  double duration_s;
  gtsam::Vector3 w;
  gtsam::Vector3 v_body;
};

/// "Typical navigation": ramps up from rest to cruise speed (as
/// RampedTwistScenario does), then drives a sequence of legs -- straight
/// cruise with occasional coordinated turns (yaw rate at constant forward
/// speed), alternating direction, the way a car follows a road with corners.
/// Each leg is an exact ConstantTwistScenario segment chained onto the
/// accumulated end pose of the previous one, so position/velocity stay
/// continuous across leg boundaries; only the body angular rate steps
/// between legs (matching the CoordinatedTurn factory's existing precedent --
/// no extra smoothing beyond the initial rest-to-cruise ramp). Past the last
/// leg the leg sequence repeats from the first leg (the ramp is not
/// repeated), each cycle chained onto the accumulated end pose of the
/// previous one, so navState(t) stays well defined -- and keeps exercising
/// turns -- for any --duration. The seam is smooth: last and first leg share
/// the same body twist (straight cruise), so position/velocity are continuous
/// across cycles just as across legs.
class TypicalNavigationScenario : public gtsam::Scenario {
 public:
  TypicalNavigationScenario(double rampDuration, double cruiseSpeed,
                            std::vector<NavLeg> legs)
      : rampDuration_(rampDuration),
        cruiseVelocityBody_(cruiseSpeed, 0.0, 0.0),
        legs_(std::move(legs)) {
    rampEndPose_ = rampPose(rampDuration_);
    double t = 0.0;     // leg times relative to ramp end
    gtsam::Pose3 pose;  // leg poses relative to ramp end
    for (const auto& leg : legs_) {
      legStartTimes_.push_back(t);
      legStartPoses_.push_back(pose);
      const gtsam::Vector6 twist =
          (gtsam::Vector6() << leg.w, leg.v_body).finished();
      pose = pose * gtsam::Pose3::Expmap(twist * leg.duration_s);
      t += leg.duration_s;
    }
    cycleDuration_ = t;
    cycleDelta_ = pose;
  }

  gtsam::Pose3 pose(double t) const override {
    if (t <= rampDuration_) {
      return rampPose(t);
    }
    size_t cycles;
    const double tau = cycleTime(t, &cycles);
    const size_t i = legIndex(tau);
    const NavLeg& leg = legs_[i];
    const gtsam::Vector6 twist =
        (gtsam::Vector6() << leg.w, leg.v_body).finished();
    gtsam::Pose3 cycleStart = rampEndPose_;
    for (size_t k = 0; k < cycles; ++k) {
      cycleStart = cycleStart * cycleDelta_;
    }
    return cycleStart * legStartPoses_[i] *
           gtsam::Pose3::Expmap(twist * (tau - legStartTimes_[i]));
  }
  gtsam::Vector3 omega_b(double t) const override {
    if (t <= rampDuration_) {
      return gtsam::Vector3::Zero();
    }
    return legs_[legIndex(cycleTime(t))].w;
  }
  gtsam::Vector3 velocity_n(double t) const override {
    if (t <= rampDuration_) {
      return rotation(t).matrix() * cruiseVelocityBody_ *
             evalRamp(t, rampDuration_).Sdot;
    }
    return rotation(t).matrix() * legs_[legIndex(cycleTime(t))].v_body;
  }
  gtsam::Vector3 acceleration_n(double t) const override {
    if (t <= rampDuration_) {
      return rotation(t) *
             (cruiseVelocityBody_ * evalRamp(t, rampDuration_).Sddot);
    }
    const NavLeg& leg = legs_[legIndex(cycleTime(t))];
    return rotation(t) * leg.w.cross(leg.v_body);
  }

 private:
  gtsam::Pose3 rampPose(double t) const {
    const gtsam::Vector6 twist =
        (gtsam::Vector6() << gtsam::Vector3::Zero(), cruiseVelocityBody_)
            .finished();
    return gtsam::Pose3::Expmap(twist * evalRamp(t, rampDuration_).S);
  }

  /// Time within the current leg cycle, in [0, cycleDuration_). Optionally
  /// reports how many full cycles precede t (for chaining cycle end poses).
  double cycleTime(double t, size_t* cycles = nullptr) const {
    const double s = t - rampDuration_;
    double n = std::floor(s / cycleDuration_);
    double tau = s - n * cycleDuration_;
    if (tau >= cycleDuration_) {  // guard the floating-point edge at a seam
      tau -= cycleDuration_;
      n += 1.0;
    }
    if (cycles) {
      *cycles = static_cast<size_t>(n);
    }
    return tau;
  }

  size_t legIndex(double tau) const {
    for (size_t i = 0; i + 1 < legs_.size(); ++i) {
      if (tau < legStartTimes_[i + 1]) {
        return i;
      }
    }
    return legs_.size() - 1;
  }

  const double rampDuration_;
  const gtsam::Vector3 cruiseVelocityBody_;
  const std::vector<NavLeg> legs_;
  gtsam::Pose3 rampEndPose_;
  double cycleDuration_ = 0.0;
  gtsam::Pose3
      cycleDelta_;  ///< end pose of one leg cycle, relative to its start
  std::vector<double> legStartTimes_;        ///< relative to ramp end
  std::vector<gtsam::Pose3> legStartPoses_;  ///< relative to ramp end
};

// --- Factories --------------------------------------------------------------

/// Static: no rotation, no motion (bias/gravity observability baseline).
inline gtsam::ConstantTwistScenario staticPose() {
  return gtsam::ConstantTwistScenario(gtsam::Vector3::Zero(),
                                      gtsam::Vector3::Zero());
}

/// Constant velocity: straight forward motion along body x, no rotation.
/// Smoothly spun up from rest to cruise speed over kRampDuration (see
/// RampedTwistScenario) so the scenario starts at v=0.
inline RampedTwistScenario constantVelocity() {
  return RampedTwistScenario(gtsam::Vector3::Zero(),
                             gtsam::Vector3(kConstSpeed, 0, 0), kRampDuration);
}

/// Pure rotation: yaw rate about body z, no translation.
inline gtsam::ConstantTwistScenario pureRotation() {
  return gtsam::ConstantTwistScenario(gtsam::Vector3(0, 0, kPureYawRate),
                                      gtsam::Vector3::Zero());
}

/// Circle: constant-twist circular path of radius kCircleRadius, smoothly spun
/// up from rest to cruise tangential speed over kRampDuration (see
/// RampedTwistScenario) -- radius/speed after the ramp exactly match the plain
/// constant-twist circle.
inline RampedTwistScenario circle() {
  const double omega = kCircleSpeed / kCircleRadius;
  return RampedTwistScenario(gtsam::Vector3(0, 0, -omega),
                             gtsam::Vector3(kCircleRadius * omega, 0, 0),
                             kRampDuration);
}

/// Coordinated turn: body yaw rate combined with a constant navigation-frame
/// acceleration (lateral + climb), starting from rest. Exercises the rotation x
/// specific-force cross-coupling that the circle (rotation only) and straight
/// line (acceleration only) scenarios miss.
inline gtsam::AcceleratingScenario coordinatedTurn() {
  const gtsam::Vector3 v0(kTurnSpeed0, 0.0,
                          0.0);  // initial nav velocity (0: starts at rest)
  const gtsam::Vector3 a_n(0.0, 0.5, 0.2);  // lateral + climb accel
  const gtsam::Vector3 omega_b(0.0, 0.0, kTurnYawRate);
  return gtsam::AcceleratingScenario(gtsam::Rot3(), gtsam::Point3(0, 0, 0), v0,
                                     a_n, omega_b);
}

/// Straight line: constant acceleration along a fixed oblique 3-D direction,
/// zero rotation, starting from rest (exercises all three velocity/position
/// channels).
inline gtsam::AcceleratingScenario straightLine() {
  const gtsam::Vector3 dir = gtsam::Vector3(2.0, 1.0, 0.5).normalized();
  const gtsam::Vector3 v0 = kLineSpeed0 * dir;  // 0: starts at rest
  const gtsam::Vector3 a_n = kLineAccel * dir;
  return gtsam::AcceleratingScenario(gtsam::Rot3(), gtsam::Point3(0, 0, 0), v0,
                                     a_n, gtsam::Vector3::Zero());
}

/// Vertical: pure vertical acceleration (accel-bias vs gravity separability
/// along the gravity axis), zero rotation, starting from rest.
inline gtsam::AcceleratingScenario vertical() {
  const gtsam::Vector3 v0(0.0, 0.0, kVertSpeed0);
  const gtsam::Vector3 a_n(0.0, 0.0, kVertAccel);
  return gtsam::AcceleratingScenario(gtsam::Rot3(), gtsam::Point3(0, 0, 0), v0,
                                     a_n, gtsam::Vector3::Zero());
}

/// Sinusoid: forward drift with a lateral oscillation (time-varying specific
/// force; a periodic consistency / NEES stressor), smoothly spun up from rest
/// over kRampDuration (see SinusoidScenario).
inline SinusoidScenario sinusoid() {
  const double omega = 2.0 * M_PI / kSinPeriod;
  return SinusoidScenario(kSinSpeed, kSinAmplitude, omega);
}

/// Default waypoint list for waypointSpline(): a weaving 3-D path (lateral
/// S-bends with a gentle climb/descent) starting at the origin, so the spline
/// exercises all three position/velocity/acceleration channels at once.
inline std::vector<gtsam::Vector3> defaultSplineWaypoints() {
  return {
      gtsam::Vector3(0.0, 0.0, 0.0),    gtsam::Vector3(10.0, 6.0, 1.0),
      gtsam::Vector3(22.0, -4.0, 2.5),  gtsam::Vector3(34.0, 5.0, 1.0),
      gtsam::Vector3(46.0, -6.0, -1.0), gtsam::Vector3(58.0, 2.0, 0.5),
      gtsam::Vector3(70.0, 0.0, 0.0),
  };
}

/// Waypoint spline: a smooth cubic path interpolated through predefined 3-D
/// waypoints, spun up from rest (see SplineScenario). Identity attitude, so it
/// stresses the translational channels and the position/DVL aiding path.
inline SplineScenario waypointSpline(
    const std::vector<gtsam::Vector3>& waypoints = defaultSplineWaypoints(),
    double pathDuration = kSplinePathDuration) {
  return SplineScenario(waypoints, pathDuration);
}

/// Typical navigation: rest -> cruise, straight legs with two occasional
/// coordinated turns in opposite directions, the way a car follows a road.
/// Sized to fill the default 30 s --duration (2 s ramp + 28 s of legs) but
/// works at any duration -- shorter runs just see a truncated route, longer
/// ones repeat the 28 s leg cycle (without the ramp), chained end to end.
/// The turns cancel (+/- kNavTurnRate for equal times), so every cycle
/// retraces the same S-bend shape translated along the original heading.
inline TypicalNavigationScenario typicalNavigation() {
  const std::vector<NavLeg> legs = {
      // Straight cruise before the first turn.
      {6.0, gtsam::Vector3::Zero(), gtsam::Vector3(kNavCruiseSpeed, 0, 0)},
      // Turn left.
      {3.0, gtsam::Vector3(0, 0, kNavTurnRate),
       gtsam::Vector3(kNavCruiseSpeed, 0, 0)},
      // Straight cruise on the new heading.
      {7.0, gtsam::Vector3::Zero(), gtsam::Vector3(kNavCruiseSpeed, 0, 0)},
      // Turn right.
      {3.0, gtsam::Vector3(0, 0, -kNavTurnRate),
       gtsam::Vector3(kNavCruiseSpeed, 0, 0)},
      // Final straight cruise, held indefinitely beyond this leg.
      {9.0, gtsam::Vector3::Zero(), gtsam::Vector3(kNavCruiseSpeed, 0, 0)},
  };
  return TypicalNavigationScenario(kRampDuration, kNavCruiseSpeed, legs);
}

}  // namespace imu_scenarios
