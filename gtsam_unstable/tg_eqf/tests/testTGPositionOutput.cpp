/**
 * @file  testTGPositionOutput.cpp
 * @brief The position output h'(xi) = R^T(pi - p), its output action, and the
 *        order of accuracy of C0 against C*.
 *
 * Mirrors testTGBodyVelocityOutput.cpp one slot over: the position output is
 * the same equivariant construction reading p instead of v, and it is the
 * channel the depth pseudo-measurement rides on.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;

using Tangent = Eigen::Matrix<double, 18, 1>;

const Eigen::Vector3d kPi(2.5, -1.0, 1.2);

bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
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

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_action {

// psi is a right action on the output space.
TEST(PositionOutput, IsARightAction) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const TGElement X = fixture::makeX(), Y = fixture::makeY();

  EXPECT(fixture::veq(
      y, PositionMeasurement::output_action(TGElement::Identity(), y)));
  EXPECT(fixture::veq(PositionMeasurement::output_action(X * Y, y),
                      PositionMeasurement::output_action(
                          Y, PositionMeasurement::output_action(X, y)),
                      fixture::kTolL));
}

// Output equivariance h'(phi(X, xi)) = psi_X(h'(xi)) with the measured position
// held fixed. This is what makes the depth channel's right-error form usable.
TEST(PositionOutput, IsEquivariant) {
  const State xi = fixture::makeXi();

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;

  for (const TGElement& X :
       {TGElement::Identity(), fixture::makeX(), TGElement::Expmap(logX)}) {
    EXPECT(fixture::veq(PositionMeasurement::predict(phi(X, xi), fixture::kPi),
                        PositionMeasurement::output_action(
                            X, PositionMeasurement::predict(xi, fixture::kPi)),
                        fixture::kTolL));
  }
}

}  // namespace output_action
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_matrix {

// C0 is the chart derivative of the output map at the reference state, with the
// measured position held fixed.
TEST(PositionOutput, C0MatchesNumerical) {
  const State xi_ref = fixture::makeXi();
  const Eigen::Matrix<double, 3, 18> H_num =
      numericalDerivative11<Eigen::Vector3d, State>(
          [](const State& x) {
            return PositionMeasurement::predict(x, fixture::kPi);
          },
          xi_ref);

  EXPECT(assert_equal(
      (Matrix)PositionMeasurement::jacobian_C0(xi_ref, fixture::kPi),
      (Matrix)H_num, 1e-5));
}

// At convergence the transported measurement is the origin output itself, so
// the C* midpoint collapses onto C0.
TEST(PositionOutput, CstarMeetsC0AtConvergence) {
  const State xi_ref = fixture::makeXi();

  // Group element whose acted position is exactly pi: p_hat = R0 g.p + p0.
  TGElement g = TGElement::Identity();
  g.p = xi_ref.R.unrotate(fixture::kPi - xi_ref.p);
  EXPECT(fixture::veq(fixture::kPi, phi(g, xi_ref).p));

  EXPECT(assert_equal(
      (Matrix)PositionMeasurement::jacobian_C0(xi_ref, fixture::kPi),
      (Matrix)PositionMeasurement::jacobian_Cstar(xi_ref, g, fixture::kPi),
      fixture::kTol));
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
namespace accuracy {

const State kXiRef = fixture::makeXi();
const TGElement kG = fixture::makeX();

/// Position the true state at eps reports to the filter.
Eigen::Vector3d measurementAt(const fixture::Tangent& eps) {
  return phi(kG, traits<State>::Retract(kXiRef, eps)).p;
}

/**
 * The residual the filter linearizes, as a function of the error.
 *
 * TGEqF::update_position feeds prediction = R0^T(pi - p_hat) against z = 0, so
 * an output matrix C is correct to the extent that
 *
 *   C eps  ~=  R0^T (p_hat - pi(eps)),
 *
 * with eps parametrising the true state as xi_true = phi(g, Retract(xi_ref,
 * eps)) and pi a noiseless reading of its position. pi moving with eps is what
 * makes this the innovation Jacobian rather than that of the output map alone.
 */
Eigen::Vector3d residual(const fixture::Tangent& eps) {
  return kXiRef.R.unrotate(phi(kG, kXiRef).p - measurementAt(eps));
}

double errorC0(const fixture::Tangent& eps) {
  return (residual(eps) -
          PositionMeasurement::jacobian_C0(kXiRef, measurementAt(eps)) * eps)
      .norm();
}

double errorCstar(const fixture::Tangent& eps) {
  return (residual(eps) -
          PositionMeasurement::jacobian_Cstar(kXiRef, kG, measurementAt(eps)) *
              eps)
      .norm();
}

/// Ratio of the linearization error at half amplitude to that at full: ~1/4 for
/// a second-order matrix, ~1/8 for a third-order one.
double halvingRatio(double (*error)(const fixture::Tangent&),
                    const fixture::Tangent& d) {
  return error(0.1 * d) / error(0.2 * d);
}

// Halving the error cuts C*'s linearization error by ~8 and C0's by ~4.
// Expanding with a = g.p and E = Exp(eps) gives residual - C* eps = 1/12
// eps^^3 a against residual - C0 eps = 1/2 eps^^2 a. This order gap is the
// reason C* exists.
TEST(PositionOutput, CstarIsThirdOrderInAttitudeError) {
  fixture::Tangent d = fixture::Tangent::Zero();
  d.segment<3>(0) = Eigen::Vector3d(0.6, -0.5, 0.62).normalized();

  // Both matrices do err at this amplitude, so the ratios are not vacuous.
  EXPECT(errorC0(0.2 * d) > 1e-4);
  EXPECT(errorCstar(0.2 * d) > 1e-8);
  EXPECT(errorCstar(0.1 * d) < errorC0(0.1 * d));

  EXPECT(halvingRatio(errorC0, d) > 0.20);  // ~1/4
  EXPECT(halvingRatio(errorC0, d) < 0.35);
  EXPECT(halvingRatio(errorCstar, d) > 0.05);  // ~1/8
  EXPECT(halvingRatio(errorCstar, d) < 0.20);
}

// Third order survives a mixed attitude-position error. This is what the
// SE_2(3) logarithm chart buys: the second-order cross term the left Jacobian
// of the exponential contributes cancels the one the C* midpoint contributes.
TEST(PositionOutput, CstarIsThirdOrderInMixedError) {
  fixture::Tangent d = fixture::Tangent::Zero();
  d.segment<3>(0) = Eigen::Vector3d(0.4, -0.3, 0.5);
  d.segment<3>(6) = Eigen::Vector3d(-0.5, 0.2, 0.35);
  d.normalize();

  EXPECT(errorCstar(0.2 * d) > 1e-8);
  EXPECT(halvingRatio(errorCstar, d) > 0.05);  // ~1/8
  EXPECT(halvingRatio(errorCstar, d) < 0.20);
  EXPECT(errorCstar(0.2 * d) < 0.5 * errorC0(0.2 * d));
}

}  // namespace accuracy
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
