/**
 * @file  testTGVirtualBiasOutput.cpp
 * @brief Unit tests for the b_v = 0 virtual-bias pseudo-measurement (Eq. B.20).
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-9;

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

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(VirtualBiasOutput, PredictReturnsVirtualBias) {
  const State xi = makeXi();
  EXPECT(veq(xi.b_v, VirtualBiasMeasurement::predict(xi)));
}

TEST(VirtualBiasOutput, PredictZeroAtIdentity) {
  EXPECT(veq(Eigen::Vector3d::Zero(),
             VirtualBiasMeasurement::predict(State::identity())));
}

TEST(VirtualBiasOutput, InnovationIsNegativePredict) {
  const State xi = makeXi();
  EXPECT(veq(-xi.b_v, VirtualBiasMeasurement::innovation(xi)));
}

// ---------------------------------------------------------------------------
// State-chart Jacobian (realises the Eq. B.20 constraint h = b_v = 0)
// ---------------------------------------------------------------------------

// Independent finite difference in the State::Retract chart. TGEqF transports
// the result to error coordinates via updateFromStateJacobian.
static Eigen::Matrix<double, 3, 18> numericalStateJacobian(const State& xi) {
  return numericalDerivative11<Eigen::Vector3d, State>(
      [](const State& x) { return VirtualBiasMeasurement::predict(x); }, xi);
}

TEST(VirtualBiasOutput, StateJacobianIsIdentityInVirtualBiasBlock) {
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, VirtualBiasMeasurement::stateJacobian(makeXi())));
  EXPECT(meq(expected,
             VirtualBiasMeasurement::stateJacobian(State::identity())));
}

TEST(VirtualBiasOutput, StateJacobianMatchesNumerical) {
  const State xi = makeXi();
  EXPECT(meq(VirtualBiasMeasurement::stateJacobian(xi),
             numericalStateJacobian(xi), 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
