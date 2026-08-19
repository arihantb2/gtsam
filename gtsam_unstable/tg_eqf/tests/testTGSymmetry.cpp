/**
 * @file  testTGSymmetry.cpp
 * @brief Action axioms for phi, and the Jacobians of the Orbit and
 *        Diffeomorphism functors in the State chart.
 *
 * Layer above testTGState.cpp and testTGGroup.cpp: how the group acts on the
 * manifold. The Jacobians here are read in the State chart, so they are the
 * first place the chart choice becomes visible.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;

bool seq(const State& a, const State& b, double tol = kTol) {
  return traits<State>::Equals(a, b, tol);
}

bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
         double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
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

/// A second state, unrelated to makeXi().
State makeXiEst() {
  State xi;
  xi.R = Rot3::Ry(-0.2) * Rot3::Rz(0.5);
  xi.v = Eigen::Vector3d(-0.7, 0.4, 1.2);
  xi.p = Eigen::Vector3d(2.0, -1.5, 0.6);
  xi.b_w = Eigen::Vector3d(-0.03, 0.01, 0.02);
  xi.b_a = Eigen::Vector3d(0.2, -0.15, 0.07);
  xi.b_v = Eigen::Vector3d(0.05, 0.0, -0.02);
  return xi;
}

TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
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

Eigen::Matrix<double, 18, 18> orbitJacobian(const TGElement& X,
                                            const State& xi_ref) {
  Eigen::Matrix<double, 18, 18> H;
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(X, &H);
  return H;
}

Eigen::Matrix<double, 18, 18> diffeoJacobian(const TGElement& X,
                                             const State& xi) {
  Eigen::Matrix<double, 18, 18> H;
  const TGSymmetry::Diffeomorphism diffeo(X);
  diffeo(xi, &H);
  return H;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace action_axioms {

// phi is a right action: the identity acts trivially and phi(XY, .) is phi(X,.)
// followed by phi(Y, .). The fiber term is what makes the order matter.
TEST(Phi, IsARightAction) {
  const State xi = fixture::makeXi();
  const TGElement X = fixture::makeX(), Y = fixture::makeY();

  EXPECT(fixture::seq(phi(TGElement::Identity(), xi), xi));
  EXPECT(fixture::seq(phi(X * Y, xi), phi(Y, phi(X, xi)), fixture::kTolL));
  EXPECT(fixture::seq(phi(X.inverse(), phi(X, xi)), xi, fixture::kTolL));
}

// On the navigation block the action is right translation of the extended pose,
// T_out = T * A. Checked in the 5x5 matrix representation, which reaches
// State::T_matrix and TGElement::to_A_matrix from an independent direction.
TEST(Phi, RightTranslatesTheExtendedPose) {
  const State xi = fixture::makeXi();
  const TGElement X = fixture::makeX();
  EXPECT(fixture::meq(phi(X, xi).T_matrix(),
                      xi.T_matrix() * X.to_A_matrix(), fixture::kTolL));
}

// The action is transitive and free: phiInverse returns the one element taking
// xi_ref to xi_est, and it recovers the element phi was called with.
TEST(PhiInverse, InvertsTheOrbitMap) {
  const State xi_ref = fixture::makeXi();
  const State xi_est = fixture::makeXiEst();
  const TGElement X = fixture::makeX();

  EXPECT(fixture::seq(phi(phiInverse(xi_ref, xi_est), xi_ref), xi_est,
                      fixture::kTolL));
  EXPECT(traits<TGElement>::Equals(phiInverse(xi_ref, phi(X, xi_ref)), X,
                                   fixture::kTolL));
}

}  // namespace action_axioms
/* ************************************************************************* */

/* ************************************************************************* */
namespace functors {

// The Orbit and Diffeomorphism functors are the two partial applications of the
// same phi, so their values must agree with it.
TEST(Symmetry, FunctorsAgreeWithPhi) {
  const State xi = fixture::makeXi();
  const TGElement X = fixture::makeX();

  const TGSymmetry::Orbit orbit(xi);
  const TGSymmetry::Diffeomorphism diffeo(X);
  EXPECT(fixture::seq(orbit(X), phi(X, xi)));
  EXPECT(fixture::seq(diffeo(xi), phi(X, xi)));
}

// Both analytic Jacobians are the true derivatives in the State chart.
TEST(Symmetry, JacobiansMatchNumerical) {
  const State xi = fixture::makeXi();
  const TGElement X = fixture::makeX();

  const Eigen::Matrix<double, 18, 18> orbit_num =
      numericalDerivative11<State, TGElement>(
          [&xi](const TGElement& x) { return phi(x, xi); }, X);
  EXPECT(fixture::meq(fixture::orbitJacobian(X, xi), orbit_num, 1e-5));

  const Eigen::Matrix<double, 18, 18> diffeo_num =
      numericalDerivative11<State, State>(
          [&X](const State& x) { return phi(X, x); }, xi);
  EXPECT(fixture::meq(fixture::diffeoJacobian(X, xi), diffeo_num, 1e-5));
}

// In the SE_2(3) logarithm chart the orbit map is right translation read in
// coordinates centred on its own output, so its navigation block is exactly the
// identity -- at every reference state and every group element, not just at the
// origin.
TEST(Orbit, NavigationJacobianIsIdentity) {
  for (const State& xi_ref : {State::identity(), fixture::makeXi()}) {
    for (const TGElement& X : {TGElement::Identity(), fixture::makeX()}) {
      EXPECT(fixture::meq(fixture::orbitJacobian(X, xi_ref).block<9, 9>(0, 0),
                          Eigen::Matrix<double, 9, 9>::Identity()));
    }
  }
}

// Differentiating phi(X^-1, phi(X, .)) = id gives back the identity, so the two
// Diffeomorphism Jacobians are mutual inverses. This constrains which side the
// adjoint acts on, which a value check at a single point does not.
TEST(Diffeomorphism, JacobianInvertsThatOfTheInverseAction) {
  const State xi = fixture::makeXi();
  const TGElement X = fixture::makeX();

  EXPECT(fixture::meq(
      fixture::diffeoJacobian(X.inverse(), phi(X, xi)) *
          fixture::diffeoJacobian(X, xi),
      Eigen::Matrix<double, 18, 18>::Identity(), fixture::kTolL));
}

}  // namespace functors
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
