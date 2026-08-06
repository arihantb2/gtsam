/**
 * @file  testTGSymmetry.cpp
 * @brief Unit tests for the TG-EqF symmetry: right action phi, the Orbit and
 * Diffeomorphism functors, and their Jacobians.
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-9;
static constexpr double kTolL = 1e-7;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool seq(const State& a, const State& b, double tol = kTol) {
  return gtsam::traits<State>::Equals(a, b, tol);
}

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

static bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

// Non-trivial state
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

// Non-trivial group element
static TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

static TGElement makeY() {
  TGElement Y;
  Y.R = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.v = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.p = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.a = {Eigen::Vector3d(0.05, 0.0, -0.05), Eigen::Vector3d(0.1, -0.1, 0.2),
         Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

// Numerical Jacobian helper: finite-difference d(phi(X, xi_ref))/dX
static Eigen::Matrix<double, 18, 18> numericalOrbitJacobian(
    const TGElement& X, const State& xi_ref) {
  return numericalDerivative11<State, TGElement>(
      [&xi_ref](const TGElement& x) { return phi(x, xi_ref); }, X);
}

// Numerical Jacobian helper: finite-difference d(phi(X, xi))/dxi
static Eigen::Matrix<double, 18, 18> numericalDiffJacobian(const TGElement& X,
                                                           const State& xi) {
  return numericalDerivative11<State, State>(
      [&X](const State& x) { return phi(X, x); }, xi);
}

// ---------------------------------------------------------------------------
// phi: basic properties
// ---------------------------------------------------------------------------

TEST(Phi, IdentityGroupElementIsNoop) {
  const TGElement I = TGElement::Identity();
  const State xi = makeXi();
  EXPECT(seq(phi(I, xi), xi));
}

// At xi = identity, the navigation output equals X's SE_2(3) part (R_X, p_X,
// v_X), and the bias equals -Ad_{C_X^{-1}}(a_X).
TEST(Phi, IdentityStateGivesGroupNavigation) {
  const TGElement X = makeX();
  const State xi0 = State::identity();
  const State result = phi(X, xi0);

  // R = I * R_X = R_X
  EXPECT(result.R.equals(X.R, kTol));
  // v = I*v_X + 0 = v_X
  EXPECT(veq(result.v, X.v));
  // p = I*p_X + 0 = p_X
  EXPECT(veq(result.p, X.p));
  // b = Ad_{A_X^{-1}}(0 - a) = -Ad_{A_X^{-1}}(a)
  const se2_3 b_expected = X.Ad_A_inv({-X.a.w, -X.a.a, -X.a.v});
  EXPECT(veq(result.b_w, b_expected.w));
  EXPECT(veq(result.b_a, b_expected.a));
  EXPECT(veq(result.b_v, b_expected.v));
}

// R_new = R_xi * R_X, p_new = R_xi*p_X + p_xi, v_new = R_xi*v_X + v_xi.
TEST(Phi, NavigationPartIsCorrect) {
  const TGElement X = makeX();
  const State xi = makeXi();
  const State result = phi(X, xi);

  EXPECT(result.R.equals(xi.R * X.R, kTol));
  EXPECT(veq(result.v, xi.R.rotate(X.v) + xi.v));
  EXPECT(veq(result.p, xi.R.rotate(X.p) + xi.p));
}

// b_new = Ad_{A_X^{-1}}(b_xi - a_X): the adjoint-shift formula.
TEST(Phi, BiasPartIsCorrect) {
  const TGElement X = makeX();
  const State xi = makeXi();
  const State result = phi(X, xi);

  const se2_3 b_diff = {xi.b_w - X.a.w, xi.b_a - X.a.a, xi.b_v - X.a.v};
  const se2_3 b_expected = X.Ad_A_inv(b_diff);
  EXPECT(veq(result.b_w, b_expected.w, kTol));
  EXPECT(veq(result.b_a, b_expected.a, kTol));
  EXPECT(veq(result.b_v, b_expected.v, kTol));
}

// ---------------------------------------------------------------------------
// phi: right group action axioms
// ---------------------------------------------------------------------------

// Navigation: (T*C_X)*C_Y = T*(C_X*C_Y) = T*C_{XY}.
// Bias: verified algebraically via the semidirect-product formula.
TEST(Phi, RightActionCompatibility) {
  const TGElement X = makeX(), Y = makeY();
  const State xi = makeXi();

  const State lhs = phi(X * Y, xi);
  const State rhs = phi(Y, phi(X, xi));

  EXPECT(seq(lhs, rhs, kTolL));
}

TEST(Phi, InverseActionUndoes) {
  const TGElement X = makeX();
  const State xi = makeXi();
  EXPECT(seq(phi(X.inverse(), phi(X, xi)), xi, kTolL));
}

// ---------------------------------------------------------------------------
// phiInverse
// ---------------------------------------------------------------------------

// A second non-trivial state, unrelated to makeXi().
static State makeXiEst() {
  State xi;
  xi.R = Rot3::Ry(-0.2) * Rot3::Rz(0.5);
  xi.v = Eigen::Vector3d(-0.7, 0.4, 1.2);
  xi.p = Eigen::Vector3d(2.0, -1.5, 0.6);
  xi.b_w = Eigen::Vector3d(-0.03, 0.01, 0.02);
  xi.b_a = Eigen::Vector3d(0.2, -0.15, 0.07);
  xi.b_v = Eigen::Vector3d(0.05, 0.0, -0.02);
  return xi;
}

// The returned element takes xi_ref to xi_est under the action.
TEST(PhiInverse, TakesRefToEst) {
  const State xi_ref = makeXi();
  const State xi_est = makeXiEst();
  EXPECT(seq(phi(phiInverse(xi_ref, xi_est), xi_ref), xi_est, kTolL));
}

// It is the exact inverse of the orbit map, so it recovers the element phi
// was called with.
TEST(PhiInverse, RecoversGroupElement) {
  const TGElement X = makeX();
  const State xi = makeXi();
  EXPECT(gtsam::traits<TGElement>::Equals(phiInverse(xi, phi(X, xi)), X, kTolL));
}

// From the identity origin the navigation part is copied straight across and
// the fiber part is -Ad_A(b_est).
TEST(PhiInverse, FromIdentityOrigin) {
  const State xi_est = makeXiEst();
  const TGElement X = phiInverse(State::identity(), xi_est);

  EXPECT(X.R.equals(xi_est.R, kTol));
  EXPECT(veq(X.v, xi_est.v));
  EXPECT(veq(X.p, xi_est.p));
  const se2_3 expected = X.Ad_A({xi_est.b_w, xi_est.b_a, xi_est.b_v});
  EXPECT(veq(X.a.w, -expected.w));
  EXPECT(veq(X.a.a, -expected.a));
  EXPECT(veq(X.a.v, -expected.v));
}

// ---------------------------------------------------------------------------
// Orbit functor
// ---------------------------------------------------------------------------

TEST(Orbit, AtIdentityReturnsRef) {
  const State xi_ref = makeXi();
  TGSymmetry::Orbit orbit(xi_ref);
  EXPECT(seq(orbit(TGElement::Identity()), xi_ref));
}

TEST(Orbit, MatchesStandalonePhiFn) {
  const State xi_ref = makeXi();
  const TGElement X = makeX();
  TGSymmetry::Orbit orbit(xi_ref);
  EXPECT(seq(orbit(X), phi(X, xi_ref)));
}

// Tolerance 1e-5 accounts for FD error.
TEST(Orbit, JacobianMatchesNumerical) {
  const State xi_ref = makeXi();
  const TGElement X = makeX();
  TGSymmetry::Orbit orbit(xi_ref);

  Eigen::Matrix<double, 18, 18> H_anal;
  orbit(X, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalOrbitJacobian(X, xi_ref);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// At (X=I, xi_ref=identity): nav block (0:9,0:9) = I_9, sigma block
// (9:18,9:18) = -I_9, cross blocks zero (ad9(0) = 0, no coupling at origin).
TEST(Orbit, JacobianAtIdentityIsSimple) {
  const State xi0 = State::identity();
  TGSymmetry::Orbit orbit(xi0);
  Eigen::Matrix<double, 18, 18> H;
  orbit(TGElement::Identity(), &H);

  EXPECT(meq(H.block<9, 9>(0, 0), Eigen::Matrix<double, 9, 9>::Identity()));
  EXPECT(meq(H.block<9, 9>(9, 0), Eigen::Matrix<double, 9, 9>::Zero()));
  EXPECT(meq(H.block<9, 9>(9, 9), -Eigen::Matrix<double, 9, 9>::Identity()));
  EXPECT(meq(H.block<9, 9>(0, 9), Eigen::Matrix<double, 9, 9>::Zero()));
}

// ---------------------------------------------------------------------------
// Diffeomorphism functor
// ---------------------------------------------------------------------------

TEST(Diffeomorphism, AtIdentityGroupIsNoop) {
  const State xi = makeXi();
  TGSymmetry::Diffeomorphism diff(TGElement::Identity());
  EXPECT(seq(diff(xi), xi));
}

TEST(Diffeomorphism, MatchesStandalonePhiFn) {
  const TGElement X = makeX();
  const State xi = makeXi();
  TGSymmetry::Diffeomorphism diff(X);
  EXPECT(seq(diff(xi), phi(X, xi)));
}

// Tolerance 1e-5 for FD error.
TEST(Diffeomorphism, JacobianMatchesNumerical) {
  const TGElement X = makeX();
  const State xi = makeXi();
  TGSymmetry::Diffeomorphism diff(X);

  Eigen::Matrix<double, 18, 18> H_anal;
  diff(xi, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalDiffJacobian(X, xi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// Since phi(I,xi)=xi, the linearisation of the identity map must be I_18.
TEST(Diffeomorphism, JacobianAtIdentityGroupIsIdentity18) {
  const State xi = makeXi();
  TGSymmetry::Diffeomorphism diff(TGElement::Identity());

  Eigen::Matrix<double, 18, 18> H;
  diff(xi, &H);

  EXPECT(meq(H, Eigen::Matrix<double, 18, 18>::Identity(), kTolL));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
