/**
 * @file  testTFGSymmetry.cpp
 * @brief Unit tests for the Two-Frames symmetry: action phi, lift Lambda,
 *        and group increment.
 *
 * Reference: Fornasier et al. (arXiv:2309.03765v3), Sec. 5.3, Theorem 6,
 *            Eq. 16-18, App. B.7.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tfg_inekf/Symmetry.h>

using namespace tfg;
using namespace gtsam;

static constexpr double kTol  = 1e-9;
static constexpr double kTolL = 1e-6;

using Vec15 = Eigen::Matrix<double, 15, 1>;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

static TwoFrameGroup makeX() {
  return TwoFrameGroup(Rot3::Rz(0.4) * Rot3::Rx(0.2),
                       Eigen::Vector3d(0.3, 0.1, -0.4),
                       Eigen::Vector3d(1.0, -2.0, 0.5),
                       Eigen::Vector3d(0.1, -0.1, 0.05),
                       Eigen::Vector3d(-0.2, 0.3, 0.0));
}

static TwoFrameGroup makeXi() {
  return TwoFrameGroup(Rot3::Ry(0.3) * Rot3::Rz(-0.1),
                       Eigen::Vector3d(0.2, -0.3, 0.1),
                       Eigen::Vector3d(-0.5, 1.0, 2.0),
                       Eigen::Vector3d(0.05, 0.0, -0.05),
                       Eigen::Vector3d(0.1, -0.1, 0.2));
}

static ImuInput makeInput() {
  ImuInput u;
  u.omega = {0.1, -0.05, 0.2};
  u.accel = {0.0, 0.0, -9.81};
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();
  return u;
}

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

// ---------------------------------------------------------------------------
// phi: right group action
// ---------------------------------------------------------------------------

// phi(I, xi) == xi: identity group element leaves state unchanged.
TEST(phi, IdentityGroupElement) {
  auto xi = makeXi();
  auto I  = TwoFrameGroup::Identity();
  EXPECT(traits<TwoFrameGroup>::Equals(phi(I, xi), xi, kTol));
}

// phi(X, I) == X: acting on the origin returns X itself.
TEST(phi, ActingOnOriginReturnsX) {
  auto X = makeX();
  auto I = TwoFrameGroup::Identity();
  EXPECT(traits<TwoFrameGroup>::Equals(phi(X, I), X, kTol));
}

// Right-action axiom (paper p.3):  phi(XY, xi) == phi(Y, phi(X, xi)).
TEST(phi, RightActionAxiom) {
  auto X = makeX(), xi = makeXi();
  // A third, distinct element so X, Y, xi are all different.
  TwoFrameGroup Y(Rot3::Rx(0.5) * Rot3::Ry(0.2), Eigen::Vector3d(-0.1, 0.4, 0.2),
                  Eigen::Vector3d(0.7, -0.3, 1.1), Eigen::Vector3d(0.0, 0.2, -0.1),
                  Eigen::Vector3d(0.3, 0.0, -0.2));
  auto lhs = phi(X * Y, xi);
  auto rhs = phi(Y, phi(X, xi));
  EXPECT(traits<TwoFrameGroup>::Equals(lhs, rhs, kTolL));
}

// ---------------------------------------------------------------------------
// lift: basic structure checks
// ---------------------------------------------------------------------------

// At identity state with zero biases and no rotation, lift with zero input
// should give only gravity in the velocity rate slot (d_v = g).
TEST(lift, GravityAtIdentity) {
  ImuInput u;
  u.omega = Eigen::Vector3d::Zero();
  u.accel = Eigen::Vector3d::Zero();  // accelerometer reads ~zero (bias-free)
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();

  auto Xi = TwoFrameGroup::Identity();
  Tangent lam = lift(Xi, u);

  // d_theta = 0 (no rotation)
  EXPECT(veq(lam.segment<3>(0), Eigen::Vector3d::Zero(), kTolL));
  // d_v = g_vec = (0, 0, -9.81)  (gravity in velocity rate)
  EXPECT(veq(lam.segment<3>(3), gravity(), kTolL));
  // d_p = v = 0  (no velocity at identity)
  EXPECT(veq(lam.segment<3>(6), Eigen::Vector3d::Zero(), kTolL));
  // d_gw, d_ga = 0  (zero biases, zero rates)
  EXPECT(veq(lam.segment<3>(9),  Eigen::Vector3d::Zero(), kTolL));
  EXPECT(veq(lam.segment<3>(12), Eigen::Vector3d::Zero(), kTolL));
}

// Position rate = current velocity  (p_dot = v, Eq. 3c).
TEST(lift, PositionRateEqualsVelocity) {
  ImuInput u;
  u.omega = Eigen::Vector3d::Zero();
  u.accel = Eigen::Vector3d::Zero();
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();

  const Eigen::Vector3d v_body(1.0, 2.0, -0.5);
  // Identity rotation so body = world
  TwoFrameGroup Xi(Rot3::Identity(), v_body, Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  Tangent lam = lift(Xi, u);
  EXPECT(veq(lam.segment<3>(6), v_body, kTolL));  // d_p = v
}

// Rotation rate = omega - b_omega  (gyro corrected for bias, Eq. 3a).
TEST(lift, RotationRateIsGyroBiasCorrected) {
  const Eigen::Vector3d omega(0.1, 0.0, 0.0);
  const Eigen::Vector3d bw(0.02, 0.0, 0.0);

  ImuInput u;
  u.omega = omega;
  u.accel = Eigen::Vector3d::Zero();
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();

  // State with known bias
  TwoFrameGroup Xi = TwoFrameGroup::FromState(Rot3::Identity(),
                                              Eigen::Vector3d::Zero(),
                                              Eigen::Vector3d::Zero(),
                                              bw,
                                              Eigen::Vector3d::Zero());
  Tangent lam = lift(Xi, u);
  EXPECT(veq(lam.segment<3>(0), omega - bw, kTolL));
}

// Bias rate (Lambda_2): with nonzero biases, d_g encodes the coupling term
// b_omega x (omega - b_omega) and b_accel x (omega - b_omega)  (Eq. 18).
TEST(lift, BiasRateEncodesCoupling) {
  const Eigen::Vector3d omega(0.1, -0.05, 0.2);
  const Eigen::Vector3d bw(0.02, -0.01, 0.03);
  const Eigen::Vector3d ba(-0.04, 0.05, 0.0);

  ImuInput u;
  u.omega = omega;
  u.accel = Eigen::Vector3d::Zero();
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();

  TwoFrameGroup Xi = TwoFrameGroup::FromState(Rot3::Identity(),
                                              Eigen::Vector3d::Zero(),
                                              Eigen::Vector3d::Zero(),
                                              bw, ba);
  Tangent lam = lift(Xi, u);

  const Eigen::Vector3d gyro_err = omega - bw;
  EXPECT(veq(lam.segment<3>(9),  bw.cross(gyro_err), kTolL));
  EXPECT(veq(lam.segment<3>(12), ba.cross(gyro_err), kTolL));
}

// Bias random walk drivers tau are subtracted from Lambda_2  (Eq. 18).
TEST(lift, TauSubtractedFromBiasRate) {
  const Eigen::Vector3d tau_w(0.001, 0.002, -0.001);
  const Eigen::Vector3d tau_a(0.003, -0.002, 0.001);

  ImuInput u_no_tau = makeInput();
  ImuInput u_tau    = makeInput();
  u_tau.tau_omega = tau_w;
  u_tau.tau_accel = tau_a;

  auto Xi      = makeXi();
  Tangent lam0 = lift(Xi, u_no_tau);
  Tangent lam1 = lift(Xi, u_tau);

  EXPECT(veq(lam0.segment<3>(9)  - tau_w, lam1.segment<3>(9),  kTolL));
  EXPECT(veq(lam0.segment<3>(12) - tau_a, lam1.segment<3>(12), kTolL));
}

// ---------------------------------------------------------------------------
// Lift reproduces the continuous dynamics (Eq. 3 / Eq. 17)
// ---------------------------------------------------------------------------

// The navigation blocks of Lambda must equal the closed form derived from
// T_dot = T Lambda_1^:
//   d_theta = omega - b_omega
//   d_v     = (accel - b_accel) + R^T g
//   d_p     = R^T v
// Checked at a generic, non-trivial state.
TEST(lift, NavigationBlocksMatchClosedForm) {
  Rot3 R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  Eigen::Vector3d v(0.3, 0.1, -0.4), p(1.0, -2.0, 0.5);
  Eigen::Vector3d bw(0.02, -0.01, 0.03), ba(-0.04, 0.05, 0.0);
  auto X = TwoFrameGroup::FromState(R, v, p, bw, ba);

  ImuInput u = makeInput();
  Tangent lam = lift(X, u);

  const Eigen::Vector3d g = gravity();
  EXPECT(veq(lam.segment<3>(0), u.omega - bw, kTolL));                  // theta
  EXPECT(veq(lam.segment<3>(3), (u.accel - ba) + R.unrotate(g), kTolL));// d_v
  EXPECT(veq(lam.segment<3>(6), R.unrotate(v), kTolL));                 // d_p
}

// One left-invariant predict step  X1 = X * Expmap(lift(X,u)*dt)  must match a
// first-order integration of the WORLD-frame dynamics (Eq. 3):
//   R1 ~ R Exp((omega - b_omega) dt)
//   v1 ~ v + (R(accel - b_accel) + g) dt
//   p1 ~ p + v dt
// to O(dt^2). Validates that the body-frame increment reproduces Eq. 3.
TEST(lift, PredictStepMatchesWorldDynamics) {
  Rot3 R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  Eigen::Vector3d v(0.3, 0.1, -0.4), p(1.0, -2.0, 0.5);
  Eigen::Vector3d bw(0.02, -0.01, 0.03), ba(-0.04, 0.05, 0.0);
  auto X = TwoFrameGroup::FromState(R, v, p, bw, ba);

  ImuInput u = makeInput();
  const double dt = 1e-5;  // small for O(dt^2) accuracy

  auto X1 = X * increment(X, u, dt);

  const Eigen::Vector3d g = gravity();
  Rot3 R1_ref = R * Rot3::Expmap((u.omega - bw) * dt);
  Eigen::Vector3d v1_ref = v + (R * (u.accel - ba) + g) * dt;
  Eigen::Vector3d p1_ref = p + v * dt;

  EXPECT(X1.R.equals(R1_ref, 1e-9));
  EXPECT(veq(X1.v, v1_ref, 1e-7));
  EXPECT(veq(X1.p, p1_ref, 1e-7));
  // Biases unchanged (tau = 0)
  EXPECT(veq(X1.bias_omega(), bw, 1e-7));
  EXPECT(veq(X1.bias_accel(), ba, 1e-7));
}

// Bias evolves by random-walk only:  with tau != 0, b changes by tau*dt.
TEST(lift, BiasRandomWalkPropagation) {
  Rot3 R = Rot3::Rz(0.2);
  Eigen::Vector3d bw(0.02, -0.01, 0.03), ba(-0.04, 0.05, 0.0);
  auto X = TwoFrameGroup::FromState(R, Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero(), bw, ba);

  ImuInput u = makeInput();
  u.tau_omega = Eigen::Vector3d(0.001, 0.0, -0.001);
  u.tau_accel = Eigen::Vector3d(0.0, 0.002, 0.0);
  const double dt = 1e-5;

  auto X1 = X * increment(X, u, dt);
  EXPECT(veq(X1.bias_omega(), bw + u.tau_omega * dt, 1e-7));
  EXPECT(veq(X1.bias_accel(), ba + u.tau_accel * dt, 1e-7));
}

// ---------------------------------------------------------------------------
// liftJacobian: matches the finite difference of lift in the right chart
// ---------------------------------------------------------------------------

// Df = d lift(X * Expmap(eps), u) / d eps must equal the central difference of
// the lift, at a generic state with non-zero biases (the regime that exercises
// every coupling block).
TEST(lift, JacobianMatchesFiniteDifference) {
  Rot3 R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  Eigen::Vector3d v(0.3, 0.1, -0.4), p(1.0, -2.0, 0.5);
  Eigen::Vector3d bw(0.02, -0.01, 0.03), ba(-0.04, 0.05, 0.0);
  auto X = TwoFrameGroup::FromState(R, v, p, bw, ba);
  ImuInput u = makeInput();

  const Eigen::Matrix<double, 15, 15> Df = liftJacobian(X, u);

  const double h = 1e-6;
  Eigen::Matrix<double, 15, 15> Df_num;
  for (int j = 0; j < 15; ++j) {
    Vec15 ep = Vec15::Zero(), em = Vec15::Zero();
    ep(j) = h;
    em(j) = -h;
    const Tangent lp = lift(traits<TwoFrameGroup>::Retract(X, ep), u);
    const Tangent lm = lift(traits<TwoFrameGroup>::Retract(X, em), u);
    Df_num.col(j) = (lp - lm) / (2 * h);
  }
  EXPECT(assert_equal((Matrix)Df, (Matrix)Df_num, 1e-6));
}

// ---------------------------------------------------------------------------
// increment: consistency with lift
// ---------------------------------------------------------------------------

// increment(I, u, dt=0) = Identity (zero step).
TEST(increment, ZeroTimeIsIdentity) {
  auto I = TwoFrameGroup::Identity();
  auto U = increment(I, makeInput(), 0.0);
  EXPECT(traits<TwoFrameGroup>::Equals(U, TwoFrameGroup::Identity(), kTol));
}

// For tiny dt, X * increment(X, u, dt) == Expmap(lift(X,u)*dt) right-composed.
TEST(increment, ConsistentWithLift) {
  auto Xi = makeXi();
  auto u  = makeInput();
  const double dt = 1e-4;

  auto U      = increment(Xi, u, dt);
  auto U_ref  = TwoFrameGroup::Expmap(lift(Xi, u) * dt);
  EXPECT(traits<TwoFrameGroup>::Equals(U, U_ref, kTol));
}

// Propagating with gravity for dt seconds increases velocity by g*dt.
TEST(increment, GravityIntegration) {
  TwoFrameGroup Xi = TwoFrameGroup::Identity();
  ImuInput u;
  u.omega = Eigen::Vector3d::Zero();
  u.accel = Eigen::Vector3d::Zero();  // accel reads zero (free-fall)
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_accel = Eigen::Vector3d::Zero();

  const double dt = 0.01;
  auto U    = increment(Xi, u, dt);
  auto Xi_1 = Xi * U;  // predict step

  // Velocity should have increased by g * dt
  EXPECT(veq(Xi_1.v, gravity() * dt, kTolL));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
