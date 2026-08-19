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
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

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
  State xi;
  xi.R = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_a = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_v = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.01, -0.02, 0.03), Eigen::Vector3d(-0.1, 0.05, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.05)};
  return X;
}

TGElement makeY() {
  TGElement Y;
  Y.R = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.v = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.p = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.a = {Eigen::Vector3d(0.05, 0.0, -0.05), Eigen::Vector3d(0.1, -0.1, 0.2),
         Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

Input makeU() {
  Input u;
  u.w = Eigen::Vector3d(0.2, -0.1, 0.05);
  u.a = Eigen::Vector3d(-0.05, 0.1, 0.0);
  u.v = Eigen::Vector3d(0.0, 0.0, 9.81);
  u.tau_w = Eigen::Vector3d(0.01, -0.02, 0.0);
  u.tau_a = Eigen::Vector3d(0.02, 0.0, 0.01);
  u.tau_v = Eigen::Vector3d(0.0, 0.03, -0.01);
  u.g_vec = Eigen::Vector3d(0.0, 0.0, -9.81);
  return u;
}

/// Pack an Input into a comparable vector, gravity included.
Eigen::Matrix<double, 21, 1> pack(const Input& u) {
  Eigen::Matrix<double, 21, 1> out;
  out << u.w, u.a, u.v, u.tau_w, u.tau_a, u.tau_v, u.g_vec;
  return out;
}

/// Lambda(phi(X, xi), psi(X, u)) - Ad_{X^-1} Lambda(xi, u), the lift condition
/// residual: zero exactly when the lift is equivariant.
Eigen::Matrix<double, 18, 1> liftConditionResidual(const TGElement& X,
                                                   const State& xi,
                                                   const Input& u) {
  const Eigen::Matrix<double, 18, 1> lhs = Lift(InputOrbit(u)(X))(phi(X, xi));
  const Eigen::Matrix<double, 18, 1> rhs =
      traits<TGElement>::AdjointMap(X.inverse()) * Lift(u)(xi);
  return lhs - rhs;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace closed_form {

// Lift::operator() evaluates a closed form. This checks both slots against the
// literal definition, Lambda_1 = (W - B + N) + T^-1 (G - N) T and
// Lambda_2 = [B, Lambda_1] - tau, built here from 5x5 matrices only.
TEST(Lift, ClosedFormMatchesMatrixConstruction) {
  const State xi = fixture::makeXi();
  const Input u = fixture::makeU();

  const Eigen::Matrix<double, 5, 5> T = xi.T_matrix();
  const se2_3 w = {u.w, u.a, u.v};
  const se2_3 b = {xi.b_w, xi.b_a, xi.b_v};
  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};
  const se2_3 g = {Eigen::Vector3d::Zero(), u.g_vec, Eigen::Vector3d::Zero()};

  Eigen::Matrix<double, 5, 5> N = Eigen::Matrix<double, 5, 5>::Zero();
  N(3, 4) = 1.0;

  const Eigen::Matrix<double, 5, 5> L1 =
      w.wedge() - b.wedge() + N + T.inverse() * (g.wedge() - N) * T;
  const Eigen::Matrix<double, 5, 5> B = b.wedge();
  const se2_3 expected_1 = se2_3::vee(L1);
  const se2_3 expected_2 = se2_3::vee(B * L1 - L1 * B - tau.wedge());

  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi);
  EXPECT(fixture::veq(expected_1.vector(), Lambda.head<9>()));
  EXPECT(fixture::veq(expected_2.vector(), Lambda.tail<9>()));
}

// The position-rate slot carries dp = v. With the virtual input cancelling its
// bias the slot is exactly R^T v.
TEST(Lift, PositionRateCarriesBodyVelocity) {
  State xi = fixture::makeXi();
  Input u = fixture::makeU();
  u.v = xi.b_v;

  EXPECT(fixture::veq(xi.R.unrotate(xi.v), Lift(u)(xi).segment<3>(6),
                      fixture::kTolL));
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

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;

  for (const TGElement& X :
       {TGElement::Identity(), fixture::makeX(), TGElement::Expmap(logX)}) {
    EXPECT(fixture::veq(Eigen::Matrix<double, 18, 1>::Zero(),
                        fixture::liftConditionResidual(X, xi, u),
                        fixture::kTolL));
  }
}

// The analytic lift differential is the true derivative in the State chart.
TEST(Lift, JacobianMatchesNumerical) {
  const State xi = fixture::makeXi();
  const Lift lift(fixture::makeU());

  Eigen::Matrix<double, 18, 18> H;
  lift(xi, &H);

  const Eigen::Matrix<double, 18, 18> H_num =
      numericalDerivative11<Eigen::Matrix<double, 18, 1>, State>(
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
  const TGElement X = fixture::makeX(), Y = fixture::makeY();

  EXPECT(fixture::veq(fixture::pack(u),
                      fixture::pack(InputOrbit(u)(TGElement::Identity()))));
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

}  // namespace input_action
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
