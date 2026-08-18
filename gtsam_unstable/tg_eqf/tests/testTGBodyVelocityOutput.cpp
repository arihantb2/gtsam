/**
 * @file  testTGBodyVelocityOutput.cpp
 * @brief Unit tests for the TG-EqF DVL body-velocity output (predict,
 * equivariance, Jacobian).
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
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
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

static Eigen::Matrix<double, 3, 18> numericalJacobian(const State& xi) {
  return numericalDerivative11<Eigen::Vector3d, State>(
      [](const State& x) { return DVLMeasurement::predict(x); }, xi);
}

static bool checkEquivariance(const TGElement& X, const State& xi) {
  const Eigen::Vector3d y = DVLMeasurement::predict(xi);
  const Eigen::Vector3d lhs = DVLMeasurement::predict(phi(X, xi));
  const Eigen::Vector3d rhs = DVLMeasurement::output_action(X, y);
  return veq(lhs, rhs, kTolL);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, PredictAtIdentityIsZero) {
  const State xi = State::identity();
  EXPECT(veq(Eigen::Vector3d::Zero(), DVLMeasurement::predict(xi)));
}

TEST(BodyVelocityOutput, PredictMatchesBodyFrameVelocity) {
  const State xi = makeXi();
  const Eigen::Vector3d expected = xi.R.unrotate(xi.v);
  EXPECT(veq(expected, DVLMeasurement::predict(xi)));
}

// ---------------------------------------------------------------------------
// output_action
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, OutputActionAtIdentityIsNoop) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  EXPECT(veq(y, DVLMeasurement::output_action(TGElement::Identity(), y)));
}

TEST(BodyVelocityOutput, OutputActionMatchesFormula) {
  const TGElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d expected = X.R.unrotate(y) + X.R.unrotate(X.v);
  EXPECT(veq(expected, DVLMeasurement::output_action(X, y)));
}

// ---------------------------------------------------------------------------
// equivariance
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, EquivarianceAtIdentity) {
  const State xi = makeXi();
  EXPECT(checkEquivariance(TGElement::Identity(), xi));
}

TEST(BodyVelocityOutput, EquivarianceForGeneralX) {
  const State xi = makeXi();
  EXPECT(checkEquivariance(makeX(), xi));
}

TEST(BodyVelocityOutput, EquivarianceForExpmapElements) {
  const State xi = makeXi();

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkEquivariance(TGElement::Expmap(logX), xi));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2, 0.05, 0.05, -0.05,
      0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkEquivariance(TGElement::Expmap(logY), xi));
}

// ---------------------------------------------------------------------------
// innovation
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, InnovationIsZMinusPredict) {
  const State xi = makeXi();
  const Eigen::Vector3d z(0.2, -0.3, 0.1);
  const Eigen::Vector3d expected = z - DVLMeasurement::predict(xi);
  EXPECT(veq(expected, DVLMeasurement::innovation(z, xi)));
}

TEST(BodyVelocityOutput, InnovationZeroWhenZMatchesPredict) {
  const State xi = makeXi();
  const Eigen::Vector3d z = DVLMeasurement::predict(xi);
  EXPECT(veq(Eigen::Vector3d::Zero(), DVLMeasurement::innovation(z, xi)));
}

// ---------------------------------------------------------------------------
// inverse_output_action
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, InverseOutputActionUndoesOutputAction) {
  const TGElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d forward = DVLMeasurement::output_action(X, y);
  EXPECT(veq(y, DVLMeasurement::inverse_output_action(X, forward)));
}

TEST(BodyVelocityOutput, InverseOutputActionMatchesFormula) {
  const TGElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d expected = X.R.rotate(y) - X.v;
  EXPECT(veq(expected, DVLMeasurement::inverse_output_action(X, y)));
}

// ---------------------------------------------------------------------------
// Jacobians
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, JacobianC0AtIdentity) {
  // Origin chart at identity: y0 = 0, so C0 = [0 | I | 0 | 0].
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, DVLMeasurement::jacobian_C0(State::identity())));
}

TEST(BodyVelocityOutput, JacobianC0MatchesDocumentedFormula) {
  const State xi_ref = makeXi();
  const Eigen::Vector3d y0 = DVLMeasurement::predict(xi_ref);

  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  expected.block<3, 3>(0, 3) = xi_ref.R.matrix().transpose();

  EXPECT(meq(expected, DVLMeasurement::jacobian_C0(xi_ref)));
}

// The output action transports the measurement exactly, so the residual the
// filter linearizes is eps -> predict(Retract(xi_ref, eps)) with the group
// estimate cancelled out. C0 is that map's chart Jacobian.
TEST(BodyVelocityOutput, JacobianC0MatchesNumerical) {
  const State xi_ref = makeXi();
  EXPECT(meq(DVLMeasurement::jacobian_C0(xi_ref), numericalJacobian(xi_ref),
             1e-5));
}

TEST(BodyVelocityOutput, JacobianCstarAtIdentityGroup) {
  const State xi_ref = makeXi();
  const Eigen::Vector3d z(0.2, -0.3, 0.1);
  // g = identity => y_tilde = z: C* = [0.5 (y0 + z)^ | R0^T | 0 | 0].
  const Eigen::Vector3d y0 = DVLMeasurement::predict(xi_ref);
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(y0 + z);
  expected.block<3, 3>(0, 3) = xi_ref.R.matrix().transpose();

  EXPECT(meq(expected, DVLMeasurement::jacobian_Cstar(
                           xi_ref, TGElement::Identity(), z)));
}

// At convergence the transported measurement is the origin output itself, so
// the midpoint collapses and C* meets C0. output_matrix_accuracy separates the
// two away from convergence instead.
TEST(BodyVelocityOutput, JacobianCstarMeetsC0AtConvergence) {
  const State xi_ref = makeXi();
  const TGElement g = makeX();
  const Eigen::Vector3d z = DVLMeasurement::predict(phi(g, xi_ref));

  EXPECT(meq(DVLMeasurement::jacobian_C0(xi_ref),
             DVLMeasurement::jacobian_Cstar(xi_ref, g, z)));
}

/* ************************************************************************* */
namespace output_matrix_accuracy {

using Tangent = Eigen::Matrix<double, 18, 1>;

const State kXiRef = makeXi();
const TGElement kG = makeX();

/// Body velocity the true state at eps reports to the filter.
Eigen::Vector3d measurementAt(const Tangent& eps) {
  return DVLMeasurement::predict(phi(kG, traits<State>::Retract(kXiRef, eps)));
}

/**
 * The residual the filter linearizes, as a function of the error.
 *
 * TGEqF::update_dvl compares predict(xi_ref) against psi_X^{-1}(z) and
 * EquivariantFilter::update forms delta_xi = -K (prediction - z), so an output
 * matrix C is correct to the extent that
 *
 *   C eps  ~=  psi_X^{-1}(z(eps)) - predict(xi_ref),
 *
 * with eps parametrising the true state as xi_true = phi(g, Retract(xi_ref,
 * eps)). The measurement moves with eps, which is what makes this the
 * innovation Jacobian rather than the Jacobian of the output map alone.
 */
Eigen::Vector3d residual(const Tangent& eps) {
  return DVLMeasurement::inverse_output_action(kG, measurementAt(eps)) -
         DVLMeasurement::predict(kXiRef);
}

/// Linearization error left by C0 at the given error.
double errorC0(const Tangent& eps) {
  return (residual(eps) - DVLMeasurement::jacobian_C0(kXiRef) * eps).norm();
}

/// Linearization error left by C* at the given error.
double errorCstar(const Tangent& eps) {
  const Eigen::Matrix<double, 3, 18> C =
      DVLMeasurement::jacobian_Cstar(kXiRef, kG, measurementAt(eps));
  return (residual(eps) - C * eps).norm();
}

// Halving a pure attitude error cuts C*'s linearization error by ~8 and C0's by
// ~4, exactly as for the position output: the residual is a rotation chord,
// and evaluating the output-action differential at the midpoint of the origin
// output and the transported measurement is the Cayley secant of that chord,
// which agrees with the tangent to third order. The two matrices are equal at
// convergence, so a test taken there cannot see this gap.
TEST(BodyVelocityOutput, CstarIsThirdOrderInAttitudeError) {
  Tangent d = Tangent::Zero();
  d.segment<3>(0) = Eigen::Vector3d(0.6, -0.5, 0.62).normalized();

  const double coarse_C0 = errorC0(0.2 * d);
  const double fine_C0 = errorC0(0.1 * d);
  const double coarse_Cstar = errorCstar(0.2 * d);
  const double fine_Cstar = errorCstar(0.1 * d);

  // Both do err at this amplitude, so the ratios below are not vacuous.
  EXPECT(coarse_C0 > 1e-4);
  EXPECT(coarse_Cstar > 1e-8);
  EXPECT(fine_Cstar < fine_C0);

  EXPECT(fine_C0 / coarse_C0 > 0.2);  // ~1/4
  EXPECT(fine_C0 / coarse_C0 < 0.35);
  EXPECT(fine_Cstar / coarse_Cstar > 0.05);  // ~1/8
  EXPECT(fine_Cstar / coarse_Cstar < 0.2);
}

// As for the position output, the third order property is chart dependent:
// traits<State>::Retract composes the rotation and adds v, so a mixed
// attitude-velocity error carries a cross term C* eps has and the true residual
// does not, and the ratio falls back to ~1/4. C* is still the smaller error of
// the two, by about 4x here.
TEST(BodyVelocityOutput, CstarDropsToSecondOrderForMixedError) {
  Tangent d = Tangent::Zero();
  d.segment<3>(0) = Eigen::Vector3d(0.4, -0.3, 0.5);
  d.segment<3>(3) = Eigen::Vector3d(-0.5, 0.2, 0.35);
  d.normalize();

  const double coarse = errorCstar(0.2 * d);
  const double fine = errorCstar(0.1 * d);

  EXPECT(coarse > 1e-5);
  EXPECT(fine / coarse > 0.2);  // ~1/4, not the ~1/8 of the pure attitude case
  EXPECT(fine / coarse < 0.35);
  EXPECT(coarse < 0.5 * errorC0(0.2 * d));
}

}  // namespace output_matrix_accuracy
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
