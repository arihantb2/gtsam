#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol  = 1e-9;
static constexpr double kTolL = 1e-7;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool seq(const TGState& a, const TGState& b, double tol = kTol) {
  return gtsam::traits<TGState>::Equals(a, b, tol);
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
static TGState makeXi() {
  TGState xi;
  xi.R       = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.v       = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.p       = Eigen::Vector3d(0.3,  0.1, -0.4);
  xi.b_w     = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_a     = Eigen::Vector3d(-0.1,  0.05, 0.0);
  xi.b_v     = Eigen::Vector3d(0.0,   0.1, -0.05);
  return xi;
}

// Non-trivial group element
static TGGroupElement makeX() {
  TGGroupElement X;
  X.R     = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v     = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p     = Eigen::Vector3d(0.3,  0.1, -0.4);
  X.a     = {Eigen::Vector3d(0.1, -0.1, 0.05),
             Eigen::Vector3d(-0.2, 0.3, 0.0),
             Eigen::Vector3d(0.0,  0.1, -0.1)};
  return X;
}

static TGGroupElement makeY() {
  TGGroupElement Y;
  Y.R     = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.v     = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.p     = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.a     = {Eigen::Vector3d(0.05, 0.0, -0.05),
             Eigen::Vector3d(0.1, -0.1, 0.2),
             Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

// Numerical Jacobian helper: finite-difference d(phi(X, xi_ref))/dX
static Eigen::Matrix<double, 18, 18> numericalOrbitJacobian(
    const TGGroupElement& X, const TGState& xi_ref, double h = 1e-6) {
  Eigen::Matrix<double, 18, 18> J;
  const TGState f0 = phi(X, xi_ref);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGGroupElement Xp = gtsam::traits<TGGroupElement>::Retract(X, e);
    const TGState fp = phi(Xp, xi_ref);
    J.col(j) = gtsam::traits<TGState>::Local(f0, fp) / h;
  }
  return J;
}

// Numerical Jacobian helper: finite-difference d(phi(X, xi))/dxi
static Eigen::Matrix<double, 18, 18> numericalDiffJacobian(
    const TGGroupElement& X, const TGState& xi, double h = 1e-6) {
  Eigen::Matrix<double, 18, 18> J;
  const TGState f0 = phi(X, xi);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xip = gtsam::traits<TGState>::Retract(xi, e);
    const TGState fp  = phi(X, xip);
    J.col(j) = gtsam::traits<TGState>::Local(f0, fp) / h;
  }
  return J;
}

// ---------------------------------------------------------------------------
// phi: basic properties
// ---------------------------------------------------------------------------

// Checks that phi(I, xi) == xi: acting with the group identity leaves the state
// unchanged (both navigation and bias).
TEST(phi, IdentityGroupElementIsNoop) {
  const TGGroupElement I = TGGroupElement::Identity();
  const TGState xi = makeXi();
  EXPECT(seq(phi(I, xi), xi));
}

// Checks phi(X, identity_state): when xi = identity, the navigation output
// equals X's SE_2(3) part (R_X, p_X, v_X), and the bias equals -Ad_{C_X^{-1}}(a_X).
TEST(phi, IdentityStateGivesGroupNavigation) {
  const TGGroupElement X = makeX();
  const TGState xi0 = TGState::identity();
  const TGState result = phi(X, xi0);

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

// Checks the navigation part of phi component-by-component:
// R_new = R_xi * R_X, p_new = R_xi*p_X + p_xi, v_new = R_xi*v_X + v_xi.
TEST(phi, NavigationPartIsCorrect) {
  const TGGroupElement X = makeX();
  const TGState xi = makeXi();
  const TGState result = phi(X, xi);

  EXPECT(result.R.equals(xi.R * X.R, kTol));
  EXPECT(veq(result.v, xi.R.rotate(X.v) + xi.v));
  EXPECT(veq(result.p, xi.R.rotate(X.p) + xi.p));
}

// Checks the bias part of phi component-by-component:
// b_new = Ad_{A_X^{-1}}(b_xi - a_X), confirming the adjoint-shift formula.
TEST(phi, BiasPartIsCorrect) {
  const TGGroupElement X = makeX();
  const TGState xi = makeXi();
  const TGState result = phi(X, xi);

  const se2_3 b_diff = {xi.b_w - X.a.w,
                        xi.b_a - X.a.a,
                        xi.b_v - X.a.v};
  const se2_3 b_expected = X.Ad_A_inv(b_diff);
  EXPECT(veq(result.b_w, b_expected.w,   kTol));
  EXPECT(veq(result.b_a, b_expected.a, kTol));
  EXPECT(veq(result.b_v, b_expected.v, kTol));
}

// ---------------------------------------------------------------------------
// phi: right group action axioms
// ---------------------------------------------------------------------------

// Checks the identity axiom phi(I, xi) == xi (formal right-action property).
TEST(phi, IdentityAction) {
  const TGState xi = makeXi();
  EXPECT(seq(phi(TGGroupElement::Identity(), xi), xi));
}

// Checks the right-action compatibility: phi(X*Y, xi) == phi(Y, phi(X, xi)).
// Navigation: (T*C_X)*C_Y = T*(C_X*C_Y) = T*C_{XY}.
// Bias: verified algebraically via the semidirect-product formula.
TEST(phi, RightActionCompatibility) {
  const TGGroupElement X = makeX(), Y = makeY();
  const TGState xi = makeXi();

  const TGState lhs = phi(X * Y, xi);
  const TGState rhs = phi(Y, phi(X, xi));

  EXPECT(seq(lhs, rhs, kTolL));
}

// Checks that applying X then X^{-1} recovers the original state:
// phi(X^{-1}, phi(X, xi)) == xi, verifying the right-action inverse property.
TEST(phi, InverseActionUndoes) {
  const TGGroupElement X = makeX();
  const TGState xi = makeXi();
  EXPECT(seq(phi(X.inverse(), phi(X, xi)), xi, kTolL));
}

// ---------------------------------------------------------------------------
// Orbit functor
// ---------------------------------------------------------------------------

// Checks that Orbit(xi_ref)(I) == xi_ref: acting with the identity returns
// the reference state unchanged.
TEST(Orbit, AtIdentityReturnsRef) {
  const TGState xi_ref = makeXi();
  TGSymmetry::Orbit orbit(xi_ref);
  EXPECT(seq(orbit(TGGroupElement::Identity()), xi_ref));
}

// Checks that Orbit(xi_ref)(X) produces the same result as the standalone
// phi(X, xi_ref), confirming the functor is a thin wrapper.
TEST(Orbit, MatchesStandalonePhiFn) {
  const TGState xi_ref = makeXi();
  const TGGroupElement X = makeX();
  TGSymmetry::Orbit orbit(xi_ref);
  EXPECT(seq(orbit(X), phi(X, xi_ref)));
}

// Checks the analytic Orbit Jacobian against a finite-difference approximation
// at a non-trivial (X, xi_ref) pair; tolerance 1e-5 accounts for FD error.
TEST(Orbit, JacobianMatchesNumerical) {
  const TGState xi_ref = makeXi();
  const TGGroupElement X = makeX();
  TGSymmetry::Orbit orbit(xi_ref);

  Eigen::Matrix<double, 18, 18> H_anal;
  orbit(X, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalOrbitJacobian(X, xi_ref);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// Checks the Orbit Jacobian at (X=I, xi_ref=identity): the nav block (0:9,0:9)
// should be I_9, the sigma block (9:18,9:18) should be -I_9, and all cross
// blocks zero (ad9(0) = 0, no coupling at the origin).
TEST(Orbit, JacobianAtIdentityIsSimple) {
  const TGState xi0 = TGState::identity();
  TGSymmetry::Orbit orbit(xi0);
  Eigen::Matrix<double, 18, 18> H;
  orbit(TGGroupElement::Identity(), &H);

  EXPECT(meq(H.block<9, 9>(0, 0), Eigen::Matrix<double, 9, 9>::Identity()));
  EXPECT(meq(H.block<9, 9>(9, 0), Eigen::Matrix<double, 9, 9>::Zero()));
  EXPECT(meq(H.block<9, 9>(9, 9), -Eigen::Matrix<double, 9, 9>::Identity()));
  EXPECT(meq(H.block<9, 9>(0, 9), Eigen::Matrix<double, 9, 9>::Zero()));
}

// ---------------------------------------------------------------------------
// Diffeomorphism functor
// ---------------------------------------------------------------------------

// Checks that Diffeomorphism(I)(xi) == xi: the map induced by the identity
// group element is the identity on M.
TEST(Diffeomorphism, AtIdentityGroupIsNoop) {
  const TGState xi = makeXi();
  TGSymmetry::Diffeomorphism diff(TGGroupElement::Identity());
  EXPECT(seq(diff(xi), xi));
}

// Checks that Diffeomorphism(X)(xi) produces the same result as phi(X, xi),
// confirming the functor delegates correctly.
TEST(Diffeomorphism, MatchesStandalonePhiFn) {
  const TGGroupElement X = makeX();
  const TGState xi = makeXi();
  TGSymmetry::Diffeomorphism diff(X);
  EXPECT(seq(diff(xi), phi(X, xi)));
}

// Checks the analytic Diffeomorphism Jacobian against a finite-difference
// approximation at a non-trivial (X, xi) pair; tolerance 1e-5 for FD error.
TEST(Diffeomorphism, JacobianMatchesNumerical) {
  const TGGroupElement X = makeX();
  const TGState xi = makeXi();
  TGSymmetry::Diffeomorphism diff(X);

  Eigen::Matrix<double, 18, 18> H_anal;
  diff(xi, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalDiffJacobian(X, xi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// Checks that Diffeomorphism(I)'s Jacobian is I_18: since phi(I,xi)=xi,
// the linearisation of the identity map must be the identity matrix.
TEST(Diffeomorphism, JacobianAtIdentityGroupIsIdentity18) {
  const TGState xi = makeXi();
  TGSymmetry::Diffeomorphism diff(TGGroupElement::Identity());

  Eigen::Matrix<double, 18, 18> H;
  diff(xi, &H);

  EXPECT(meq(H, Eigen::Matrix<double, 18, 18>::Identity(), kTolL));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
