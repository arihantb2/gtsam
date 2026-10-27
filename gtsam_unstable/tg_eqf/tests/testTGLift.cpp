/**
 * @file  testTGLift.cpp
 * @brief The equivariant lift Lambda and the input action psi.
 *
 * Layer above testTGSymmetry.cpp: the system dynamics carried onto the group.
 * The two claims that matter are the closed form against the literal matrix
 * construction, and the lift condition that makes the filter equivariant.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/TGEqF.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;

bool veq(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
         double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

State makeXi() {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = Vector3(1.0, -2.0, 0.5);
  vp.col(1) = Vector3(0.3, 0.1, -0.4);
  State xi;
  xi.T = ExtendedPose3<2>(Rot3::Rz(0.3) * Rot3::Rx(0.1), vp);
  xi.b = (Vector9() << 0.01, -0.02, 0.03, -0.1, 0.05, 0.0, 0.0, 0.1, -0.05)
             .finished();
  xi.bm = Vector3(0.3, -0.2, 0.4);
  return xi;
}

TGMagElement makeElement(const Rot3& R, const Vector3& v, const Vector3& p,
                         const Vector9& a, const Vector3& d) {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = v;
  vp.col(1) = p;
  return TGMagElement(TGElement(ExtendedPose3<2>(R, vp), a), d);
}

TGMagElement makeX() {
  return makeElement(
      Rot3::Rz(0.4) * Rot3::Rx(0.2), Vector3(1.0, -2.0, 0.5),
      Vector3(0.3, 0.1, -0.4),
      (Vector9() << 0.01, -0.02, 0.03, -0.1, 0.05, 0.0, 0.0, 0.1, -0.05)
          .finished(),
      Vector3(0.1, 0.2, -0.3));
}

TGMagElement makeY() {
  return makeElement(
      Rot3::Ry(0.3) * Rot3::Rz(-0.1), Vector3(-0.5, 1.0, 2.0),
      Vector3(0.2, -0.3, 0.1),
      (Vector9() << 0.05, 0.0, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.05)
          .finished(),
      Vector3(-0.2, 0.05, 0.15));
}

Input makeU() {
  Input u;
  u.w = Vector3(0.2, -0.1, 0.05);
  u.a = Vector3(-0.05, 0.1, 0.0);
  u.v = Vector3(0.0, 0.0, 9.81);
  u.tau_w = Vector3(0.01, -0.02, 0.0);
  u.tau_a = Vector3(0.02, 0.0, 0.01);
  u.tau_v = Vector3(0.0, 0.03, -0.01);
  u.tau_m = Vector3(-0.01, 0.02, 0.03);
  u.g_vec = Vector3(0.0, 0.0, -9.81);
  return u;
}

/// Pack an Input into a comparable vector, gravity included.
Input::Vector24 pack(const Input& u) {
  Input::Vector24 out;
  out << u.w, u.a, u.v, u.tau_w, u.tau_a, u.tau_v, u.tau_m, u.g_vec;
  return out;
}

using Tangent = State::TangentVector;

/// Lambda(phi(X, xi), psi(X, u)) - Ad_{X^-1} Lambda(xi, u), the lift condition
/// residual: zero exactly when the lift is equivariant.
Tangent liftConditionResidual(const TGMagElement& X, const State& xi,
                              const Input& u) {
  const Tangent lhs = Lift(InputOrbit(u)(X))(phi(X, xi));
  const Tangent rhs =
      traits<TGMagElement>::AdjointMap(X.inverse()) * Lift(u)(xi);
  return lhs - rhs;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace closed_form {

// Lift::operator() evaluates a closed form. This checks the tangent-group slots
// against the literal definition, Lambda_1 = (W - B + N) + T^-1 (G - N) T and
// Lambda_2 = [B, Lambda_1] - tau, built here from 5x5 matrices only, and the
// magnetometer slot against Lambda_3 = bm^ (w - b_w) - tau_m.
TEST(Lift, ClosedFormMatchesMatrixConstruction) {
  const State xi = fixture::makeXi();
  const Input u = fixture::makeU();

  using Se23 = ExtendedPose3<2>;

  const Eigen::Matrix<double, 5, 5> T = xi.T_matrix();
  Vector9 w;
  w.segment<3>(0) = u.w;
  w.segment<3>(3) = u.a;
  w.segment<3>(6) = u.v;
  const Vector9& b = xi.b;
  Vector9 tau;
  tau.segment<3>(0) = u.tau_w;
  tau.segment<3>(3) = u.tau_a;
  tau.segment<3>(6) = u.tau_v;
  Vector9 g = Vector9::Zero();
  g.segment<3>(3) = u.g_vec;

  Eigen::Matrix<double, 5, 5> N = Eigen::Matrix<double, 5, 5>::Zero();
  N(3, 4) = 1.0;

  const Eigen::Matrix<double, 5, 5> L1 =
      Se23::Hat(w) - Se23::Hat(b) + N + T.inverse() * (Se23::Hat(g) - N) * T;
  const Eigen::Matrix<double, 5, 5> B = Se23::Hat(b);
  const Vector9 expected_1 = Se23::Vee(L1);
  const Vector9 expected_2 = Se23::Vee(B * L1 - L1 * B - Se23::Hat(tau));

  const fixture::Tangent Lambda = Lift(u)(xi);
  EXPECT(fixture::veq(expected_1, Lambda.segment<9>(0)));
  EXPECT(fixture::veq(expected_2, Lambda.segment<9>(9)));
  EXPECT(fixture::veq(skewSymmetric(xi.bm) * (u.w - xi.b.head<3>()) - u.tau_m,
                      Lambda.segment<3>(18)));
}

// The lift generates the true dynamics through the action. For the
// magnetometer bias that is bm_dot = tau_m, read here as the finite-difference
// rate of phi(Exp(t Lambda), xi).bm.
TEST(Lift, ReproducesTheMagBiasRandomWalk) {
  const State xi = fixture::makeXi();
  const Input u = fixture::makeU();
  const fixture::Tangent Lambda = Lift(u)(xi);

  const double h = 1e-6;
  const Vector3 rate = (phi(TGMagElement::Expmap(h * Lambda), xi).bm -
                        phi(TGMagElement::Expmap(-h * Lambda), xi).bm) /
                       (2.0 * h);
  EXPECT(fixture::veq(u.tau_m, rate, 1e-8));
}

// The position-rate slot carries dp = v. With the virtual input cancelling its
// bias the slot is exactly R^T v.
TEST(Lift, PositionRateCarriesBodyVelocity) {
  State xi = fixture::makeXi();
  Input u = fixture::makeU();
  u.v = xi.b.segment<3>(6);

  EXPECT(fixture::veq(xi.T.rotation().unrotate(xi.T.x(0)),
                      Lift(u)(xi).segment<3>(6), fixture::kTolL));
}

}  // namespace closed_form
/* ************************************************************************* */

/* ************************************************************************* */
namespace equivariance {

// The lift condition, at the identity and at group elements reached both by
// hand and through Expmap. This is what lets the filter linearize once at the
// origin instead of tracking the estimate.
TEST(Lift, SatisfiesTheLiftCondition) {
  const State xi = fixture::makeXi();
  const Input u = fixture::makeU();

  fixture::Tangent logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1, 0.2, -0.3, 0.1;

  for (const TGMagElement& X : {TGMagElement::Identity(), fixture::makeX(),
                                TGMagElement::Expmap(logX)}) {
    EXPECT(fixture::veq(fixture::Tangent::Zero(),
                        fixture::liftConditionResidual(X, xi, u),
                        fixture::kTolL));
  }
}

// The analytic lift differential is the true derivative in the State chart,
// including the magnetometer rows at a nonzero bm.
TEST(Lift, JacobianMatchesNumerical) {
  const State xi = fixture::makeXi();
  const Lift lift(fixture::makeU());

  State::Jacobian H;
  lift(xi, &H);

  const State::Jacobian H_num = numericalDerivative11<fixture::Tangent, State>(
      [&lift](const State& x) { return lift(x); }, xi);
  EXPECT(assert_equal((Matrix)H, (Matrix)H_num, 1e-5));
}

}  // namespace equivariance
/* ************************************************************************* */

/* ************************************************************************* */
namespace input_action {

// psi is a right action on the input space, the axiom that makes the lift
// condition composable across group elements.
TEST(InputOrbit, IsARightAction) {
  const Input u = fixture::makeU();
  const TGMagElement X = fixture::makeX(), Y = fixture::makeY();

  EXPECT(fixture::veq(fixture::pack(u),
                      fixture::pack(InputOrbit(u)(TGMagElement::Identity()))));
  EXPECT(fixture::veq(fixture::pack(InputOrbit(u)(X * Y)),
                      fixture::pack(InputOrbit(InputOrbit(u)(X))(Y)),
                      fixture::kTolL));
}

// Gravity is a fixed frame quantity, not part of the input the group acts on,
// so it must pass through untouched.
TEST(InputOrbit, LeavesGravityUntouched) {
  const Input u = fixture::makeU();
  EXPECT(fixture::veq(u.g_vec, InputOrbit(u)(fixture::makeX()).g_vec));
}

// The bias random walk is a body-frame rate, so it only rotates: tau_m ->
// R_A^T tau_m, independent of d.
TEST(InputOrbit, RotatesTheMagBiasRandomWalk) {
  const Input u = fixture::makeU();
  const TGMagElement X = fixture::makeX();
  EXPECT(fixture::veq(X.first.first.rotation().unrotate(u.tau_m),
                      InputOrbit(u)(X).tau_m));
}

}  // namespace input_action
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
