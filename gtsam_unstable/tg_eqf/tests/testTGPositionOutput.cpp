/**
 * @file  testTGPositionOutput.cpp
 * @brief Unit tests for the TG-EqF position output (predict, equivariance,
 * C0/C* Jacobians; App. B.19).

 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
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

static Eigen::Matrix<double, 3, 18> numericalJacobian(
    const State& xi, const Eigen::Vector3d& pi) {
  return numericalDerivative11<Eigen::Vector3d, State>(
      [&pi](const State& x) { return PositionMeasurement::predict(x, pi); },
      xi);
}

static bool checkEquivariance(const TGElement& X, const State& xi,
                              const Eigen::Vector3d& pi) {
  const Eigen::Vector3d y = PositionMeasurement::predict(xi, pi);
  const Eigen::Vector3d lhs = PositionMeasurement::predict(phi(X, xi), pi);
  const Eigen::Vector3d rhs = PositionMeasurement::output_action(X, y);
  return veq(lhs, rhs, kTolL);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(PositionOutput, PredictAtIdentityIsZeroWhenPiEqualsP) {
  State xi = State::identity();
  const Eigen::Vector3d pi(1.0, -2.0, 0.5);
  xi.p = pi;
  EXPECT(veq(Eigen::Vector3d::Zero(), PositionMeasurement::predict(xi, pi)));
}

TEST(PositionOutput, PredictMatchesBodyFrameResidual) {
  const State xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Vector3d expected = xi.R.unrotate(pi - xi.p);
  EXPECT(veq(expected, PositionMeasurement::predict(xi, pi)));
}

// ---------------------------------------------------------------------------
// output_action
// ---------------------------------------------------------------------------

TEST(PositionOutput, OutputActionAtIdentityIsNoop) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  EXPECT(veq(y, PositionMeasurement::output_action(TGElement::Identity(), y)));
}

TEST(PositionOutput, OutputActionMatchesFormula) {
  const TGElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d expected = X.R.unrotate(y - X.p);
  EXPECT(veq(expected, PositionMeasurement::output_action(X, y)));
}

// ---------------------------------------------------------------------------
// equivariance
// ---------------------------------------------------------------------------

TEST(PositionOutput, EquivarianceAtIdentity) {
  const State xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  EXPECT(checkEquivariance(TGElement::Identity(), xi, pi));
}

TEST(PositionOutput, EquivarianceForGeneralX) {
  const State xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  EXPECT(checkEquivariance(makeX(), xi, pi));
}

TEST(PositionOutput, EquivarianceForExpmapElements) {
  const State xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkEquivariance(TGElement::Expmap(logX), xi, pi));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2, 0.05, 0.05, -0.05,
      0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkEquivariance(TGElement::Expmap(logY), xi, pi));
}

// ---------------------------------------------------------------------------
// innovation
// ---------------------------------------------------------------------------

TEST(PositionOutput, InnovationIsNegativePredict) {
  const State xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Vector3d expected = -PositionMeasurement::predict(xi, pi);
  EXPECT(veq(expected, PositionMeasurement::innovation(pi, xi)));
}

TEST(PositionOutput, InnovationZeroWhenPiEqualsP) {
  const State xi = makeXi();
  const Eigen::Vector3d pi = xi.p;
  EXPECT(veq(Eigen::Vector3d::Zero(), PositionMeasurement::innovation(pi, xi)));
}

// ---------------------------------------------------------------------------
// Jacobians
// ---------------------------------------------------------------------------

// Origin-chart finite difference of  eps -> R0^T(pi - p(phi(g, Retract(xi_ref,
// eps)))) with the measurement pi held fixed. This is the Jacobian of the
// output map, not of the innovation the filter consumes: pi is a reading of the
// position and therefore moves with eps. See the output_matrix_accuracy fixture
// for a check on the matrix the filter actually needs.
static Eigen::Matrix<double, 3, 18> numericalOriginJacobian(
    const State& xi_ref, const TGElement& g, const Eigen::Vector3d& pi,
    double h = 1e-6) {
  auto m = [&](const State& xr) {
    return xi_ref.R.unrotate(pi - phi(g, xr).p);
  };
  Eigen::Matrix<double, 3, 18> J;
  const Eigen::Vector3d f0 = m(xi_ref);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    J.col(j) = (m(traits<State>::Retract(xi_ref, e)) - f0) / h;
  }
  return J;
}

TEST(PositionOutput, JacobianC0AtIdentity) {
  const State xi_ref = State::identity();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  // Origin chart at identity: C0 = [skew(pi) | 0 | -I | 0].
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = gtsam::skewSymmetric(pi);
  expected.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, PositionMeasurement::jacobian_C0(xi_ref, pi)));
}

TEST(PositionOutput, JacobianC0MatchesNumericalAtIdentity) {
  // At the identity origin with g = identity the origin and estimate charts
  // coincide, so C0 equals the full chart finite difference of predict.
  const State xi_ref = State::identity();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Matrix<double, 3, 18> H_num = numericalJacobian(xi_ref, pi);
  const Eigen::Matrix<double, 3, 18> H_anal =
      PositionMeasurement::jacobian_C0(xi_ref, pi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

TEST(PositionOutput, JacobianCstarAtIdentityWithPi) {
  const State xi_ref = State::identity();
  const TGElement g = TGElement::Identity();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  // g = identity => p_X = 0, y0 = pi: C* = [0.5 skew(pi) | 0 | -I | 0].
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(pi);
  expected.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, PositionMeasurement::jacobian_Cstar(xi_ref, g, pi)));
}

// At convergence C0 and C* coincide, so this pins neither against the other;
// output_matrix_accuracy separates them by order of accuracy instead.
TEST(PositionOutput, JacobianCstarMatchesOriginFDAtConvergence) {
  const State xi_ref = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);

  // Group whose acted position equals pi (convergence): p_hat = R0*g.p + p0.
  TGElement g = TGElement::Identity();
  g.p = xi_ref.R.unrotate(pi - xi_ref.p);  // = y0
  // sanity: phi(g, xi_ref).p == pi
  EXPECT(veq(pi, phi(g, xi_ref).p, 1e-9));

  const Eigen::Matrix<double, 3, 18> H_anal =
      PositionMeasurement::jacobian_Cstar(xi_ref, g, pi);
  const Eigen::Matrix<double, 3, 18> H_num =
      numericalOriginJacobian(xi_ref, g, pi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

/* ************************************************************************* */
namespace output_matrix_accuracy {

using Tangent = Eigen::Matrix<double, 18, 1>;

const State kXiRef = makeXi();
const TGElement kG = makeX();

/**
 * The residual the filter linearizes, as a function of the error.
 *
 * TGEqF::update_position feeds prediction = R0^T(pi - p_hat) against z = 0, and
 * EquivariantFilter::update forms delta_xi = -K (prediction - z). An output
 * matrix C is therefore correct to the extent that
 *
 *   C eps  ~=  R0^T (p_hat - pi(eps)),
 *
 * where eps parametrises the true state as xi_true = phi(g, Retract(xi_ref,
 * eps)) and pi is a noiseless reading of its position. pi moving with eps is
 * what makes this the innovation Jacobian rather than the Jacobian of the
 * output map with pi frozen, which is what numericalOriginJacobian computes.
 */
Eigen::Vector3d residual(const Tangent& eps) {
  const Eigen::Vector3d p_hat = phi(kG, kXiRef).p;
  const Eigen::Vector3d pi = phi(kG, traits<State>::Retract(kXiRef, eps)).p;
  return kXiRef.R.unrotate(p_hat - pi);
}

/// Position the true state at eps reports to the filter.
Eigen::Vector3d measurementAt(const Tangent& eps) {
  return phi(kG, traits<State>::Retract(kXiRef, eps)).p;
}

/// Linearization error left by C0 at the given error.
double errorC0(const Tangent& eps) {
  const Eigen::Matrix<double, 3, 18> C =
      PositionMeasurement::jacobian_C0(kXiRef, measurementAt(eps));
  return (residual(eps) - C * eps).norm();
}

/// Linearization error left by C* at the given error.
double errorCstar(const Tangent& eps) {
  const Eigen::Matrix<double, 3, 18> C =
      PositionMeasurement::jacobian_Cstar(kXiRef, kG, measurementAt(eps));
  return (residual(eps) - C * eps).norm();
}

// Halving a pure attitude error cuts C*'s linearization error by ~8 and C0's by
// ~4: C* is third order and C0 second. Expanding with a = g.p and E = Exp(eps^)
// gives residual - C* eps = 1/12 eps^^3 a and residual - C0 eps = 1/2 eps^^2 a.
// This order gap is the whole point of Equ. (B.19); the two matrices are equal
// at convergence, so a test taken there cannot see it.
TEST(PositionOutput, CstarIsThirdOrderInAttitudeError) {
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

  EXPECT(fine_C0 / coarse_C0 > 0.2);        // ~1/4
  EXPECT(fine_C0 / coarse_C0 < 0.35);
  EXPECT(fine_Cstar / coarse_Cstar > 0.05);  // ~1/8
  EXPECT(fine_Cstar / coarse_Cstar < 0.2);
}

// The third order property is chart dependent and this chart loses it.
// Fornasier et al. derive Equ. (B.19) in the SE_2(3) logarithm chart, whereas
// traits<State>::Retract composes the rotation and adds v and p, so the true
// residual carries no attitude-position cross term while C* eps does: with
// q = R0^T eps_p the leftover is -1/2 q x eps_R, second order. Moving the state
// chart to the SE_2(3) exponential should turn this ratio into ~1/8.
TEST(PositionOutput, CstarDropsToSecondOrderForMixedError) {
  Tangent d = Tangent::Zero();
  d.segment<3>(0) = Eigen::Vector3d(0.4, -0.3, 0.5);
  d.segment<3>(6) = Eigen::Vector3d(-0.5, 0.2, 0.35);
  d.normalize();

  const double coarse = errorCstar(0.2 * d);
  const double fine = errorCstar(0.1 * d);

  EXPECT(coarse > 1e-5);
  EXPECT(fine / coarse > 0.2);  // ~1/4, not the ~1/8 of the pure attitude case
  EXPECT(fine / coarse < 0.35);
}

}  // namespace output_matrix_accuracy
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
