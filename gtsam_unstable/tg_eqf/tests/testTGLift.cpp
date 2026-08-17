/**
 * @file  testTGLift.cpp
 * @brief Unit tests for the TG-EqF lift Lambda (Theorem 9) and InputOrbit.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-9;
static constexpr double kTolL = 1e-7;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

static bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

static State makeXi() {
  State xi;
  xi.R = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_a = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_v = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

static TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.01, -0.02, 0.03), Eigen::Vector3d(-0.1, 0.05, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.05)};
  return X;
}

static Input makeU() {
  Input u;
  u.w = Eigen::Vector3d(0.2, -0.1, 0.05);
  u.a = Eigen::Vector3d(-0.05, 0.1, 0.0);
  u.v = Eigen::Vector3d(0.0, 0.0, 9.81);
  u.tau_w = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();
  u.g_vec = Eigen::Vector3d(0.0, 0.0, -9.81);
  return u;
}

static Eigen::Matrix<double, 18, 18> numericalLiftJacobian(const Lift& lift,
                                                           const State& xi) {
  return numericalDerivative11<Eigen::Matrix<double, 18, 1>, State>(
      [&lift](const State& x) { return lift(x); }, xi);
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

TEST(Input, VectorRoundTrip) {
  const Input u = makeU();
  const Input recovered = Input::from_vector(u.vector());
  EXPECT(veq(u.w, recovered.w));
  EXPECT(veq(u.a, recovered.a));
  EXPECT(veq(u.v, recovered.v));
  EXPECT(veq(u.tau_w, recovered.tau_w));
  EXPECT(veq(u.tau_a, recovered.tau_a));
  EXPECT(veq(u.tau_v, recovered.tau_v));
  EXPECT(veq(u.g_vec, recovered.g_vec));
}

// ---------------------------------------------------------------------------
// Closed-form Lambda_1 vs. the original wedge/matrix construction
// ---------------------------------------------------------------------------

// Lift::operator() computes Lambda_1 via the closed form derived in Lift.h
// (R^T g in the "a" slot, R^T v in the "v" slot). This pins it against the
// literal Theorem 9 construction Lambda_1 = (W-B+N) + T^{-1}(G-N)T, built
// here from public se2_3::wedge() calls, at a general (non-identity) state.
TEST(Lift, ClosedFormMatchesMatrixConstruction) {
  const State xi = makeXi();
  const Input u = makeU();

  const Eigen::Matrix<double, 5, 5> T = xi.T_matrix();
  const se2_3 w = {u.w, u.a, u.v};
  const se2_3 b = {xi.b_w, xi.b_a, xi.b_v};
  const se2_3 g = {Eigen::Vector3d::Zero(), u.g_vec, Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 5, 5> N = Eigen::Matrix<double, 5, 5>::Zero();
  N(3, 4) = 1.0;
  const Eigen::Matrix<double, 5, 5> L1_mat =
      w.wedge() - b.wedge() + N + T.inverse() * (g.wedge() - N) * T;
  const se2_3 Lambda1_expected = se2_3::vee(L1_mat);

  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi);
  EXPECT(assert_equal((Vector)Lambda1_expected.vector(),
                      (Vector)Lambda.head<9>(), 1e-9));
}

// ---------------------------------------------------------------------------
// Lift value at origin
// ---------------------------------------------------------------------------

TEST(Lift, AtIdentityStateLambda1IsWPlusG) {
  const Input u = makeU();
  const State xi0 = State::identity();
  Lift lift(u);

  // At T=I, b=0: Lambda_1 = W + G (N cancels in (W-B+N) + (G-N)).
  // Gravity sits in the velocity ("a") slot: vdot = Ra + g (Eq. 3b).
  const se2_3 w = {u.w, u.a, u.v};
  const se2_3 g = {Eigen::Vector3d::Zero(), u.g_vec, Eigen::Vector3d::Zero()};
  const se2_3 expected = se2_3::vee(w.wedge() + g.wedge());
  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi0);

  EXPECT(veq(expected.w, Lambda.segment<3>(0), kTolL));
  EXPECT(veq(expected.a, Lambda.segment<3>(3), kTolL));
  EXPECT(veq(expected.v, Lambda.segment<3>(6), kTolL));
}

TEST(Lift, AtIdentityStateLambda2IsMinusTau) {
  Input u = makeU();
  u.tau_w = Eigen::Vector3d(0.01, -0.02, 0.0);
  u.tau_a = Eigen::Vector3d(0.02, 0.0, 0.01);
  u.tau_v = Eigen::Vector3d(0.0, 0.03, -0.01);

  const State xi0 = State::identity();
  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi0);

  Eigen::Matrix<double, 9, 1> tau_vec;
  tau_vec.segment<3>(0) = u.tau_w;
  tau_vec.segment<3>(3) = u.tau_a;
  tau_vec.segment<3>(6) = u.tau_v;
  EXPECT(assert_equal((Vector)(-tau_vec), (Vector)Lambda.tail<9>(), kTolL));
}

// Regression: the lift must encode the position kinematics dp = v. The
// position-rate slot of Lambda_1 is R^T(nu - b_v) + R^T v; this checks the lift
// identity with nu = b_v (slot == R^T v). The filter itself feeds nu = 0
// and pins b_v ~ 0 with the default anchor, so at its operating
// point the slot is also ~ R^T v. A constant-N lift drops this and freezes
// position.
TEST(Lift, PositionRateEncodesBodyVelocity) {
  State xi = makeXi();
  Input u = makeU();
  u.v = xi.b_v;  // virtual velocity input cancels its bias

  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi);
  const Eigen::Vector3d body_v = xi.R.unrotate(xi.v);
  EXPECT(veq(body_v, Lambda.segment<3>(6), kTolL));  // v slot == R^T v
}

// The position-rate slot is R^T v + (nu - b_v): the virtual input enters the
// v-slot of W directly and the bias via B. With nu = 0 (the filter's choice)
// it is R^T v - b_v, equal to R^T v only when b_v = 0 (kept by the anchor).
TEST(Lift, PositionRateWithZeroVirtualInput) {
  State xi = makeXi();
  Input u = makeU();
  u.v = Eigen::Vector3d::Zero();  // nu = 0

  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi);
  const Eigen::Vector3d expected = xi.R.unrotate(xi.v) - xi.b_v;
  EXPECT(veq(expected, Lambda.segment<3>(6), kTolL));
}

// ---------------------------------------------------------------------------
// Lift equivariance
// ---------------------------------------------------------------------------

TEST(Lift, EquivarianceAtIdentity) {
  const Input u = makeU();
  const State xi = makeXi();
  const TGElement I = TGElement::Identity();
  Lift lift(u);

  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi);
  const Eigen::Matrix<double, 18, 1> lhs = Lift(InputOrbit(u)(I))(phi(I, xi));
  const Eigen::Matrix<double, 18, 1> rhs =
      traits<TGElement>::AdjointMap(I.inverse()) * Lambda;

  EXPECT(assert_equal((Vector)lhs, (Vector)rhs, kTolL));
  EXPECT(assert_equal((Vector)lhs, (Vector)Lambda, kTolL));
}

static bool checkLiftEquivariance(const TGElement& X, const State& xi,
                                  const Input& u) {
  Lift lift(u);
  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi);
  const Eigen::Matrix<double, 18, 1> lhs = Lift(InputOrbit(u)(X))(phi(X, xi));
  const Eigen::Matrix<double, 18, 1> rhs =
      traits<TGElement>::AdjointMap(X.inverse()) * Lambda;
  return assert_equal((Vector)lhs, (Vector)rhs, kTolL);
}

TEST(Lift, EquivarianceForGeneralX) {
  const Input u = makeU();
  const State xi = makeXi();
  EXPECT(checkLiftEquivariance(makeX(), xi, u));
}

TEST(Lift, EquivarianceForExpmapElements) {
  const Input u = makeU();
  const State xi = makeXi();

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkLiftEquivariance(TGElement::Expmap(logX), xi, u));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2, 0.05, 0.05, -0.05,
      0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkLiftEquivariance(TGElement::Expmap(logY), xi, u));
}

// ---------------------------------------------------------------------------
// Lift Jacobian
// ---------------------------------------------------------------------------

TEST(Lift, JacobianMatchesNumerical) {
  const Input u = makeU();
  const State xi = makeXi();
  Lift lift(u);

  Eigen::Matrix<double, 18, 18> H_anal;
  lift(xi, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalLiftJacobian(lift, xi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// ---------------------------------------------------------------------------
// InputOrbit
// ---------------------------------------------------------------------------

TEST(InputOrbit, AtIdentityIsNoop) {
  const Input u = makeU();
  const Input result = InputOrbit(u)(TGElement::Identity());
  EXPECT(veq(u.w, result.w));
  EXPECT(veq(u.a, result.a));
  EXPECT(veq(u.v, result.v));
  EXPECT(veq(u.tau_w, result.tau_w));
  EXPECT(veq(u.tau_a, result.tau_a));
  EXPECT(veq(u.tau_v, result.tau_v));
  EXPECT(veq(u.g_vec, result.g_vec));
}

TEST(InputOrbit, GravityUnchanged) {
  const Input u = makeU();
  const Input result = InputOrbit(u)(makeX());
  EXPECT(veq(u.g_vec, result.g_vec));
}

TEST(InputOrbit, TauTransformedByAdInv) {
  const Input u = makeU();
  const TGElement X = makeX();
  const Input result = InputOrbit(u)(X);

  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};
  const se2_3 tau_expected = X.Ad_A_inv(tau);
  EXPECT(veq(result.tau_w, tau_expected.w, kTolL));
  EXPECT(veq(result.tau_a, tau_expected.a, kTolL));
  EXPECT(veq(result.tau_v, tau_expected.v, kTolL));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
