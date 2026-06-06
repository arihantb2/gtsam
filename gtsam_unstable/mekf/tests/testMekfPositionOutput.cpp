/**
 * @file  testMekfPositionOutput.cpp
 * @brief Unit tests for the naive world-frame position output (h=p, H, FD check).
 *
 * Block order = [ d_theta(3) | d_v(3) | d_p(3) | d_b_gyro(3) | d_b_accel(3) ].
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/mekf/PositionOutput.h>

using namespace mekf;
using namespace gtsam;

static constexpr double kTol = 1e-9;

using Tangent = Eigen::Matrix<double, 15, 1>;

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}
static bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

static MekfState exampleState() {
  return MekfState(Rot3::Rz(0.4) * Rot3::Rx(0.2), Eigen::Vector3d(1.0, -0.5, 0.3),
                   Eigen::Vector3d(2.0, 1.0, -1.0),
                   Eigen::Vector3d(0.01, -0.02, 0.03),
                   Eigen::Vector3d(-0.05, 0.1, 0.0));
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

// h(xi) = p (the global position), independent of attitude / velocity / bias.
TEST(MekfPositionOutput, PredictReturnsPosition) {
  const MekfState X = exampleState();
  EXPECT(veq(PositionOutput::predict(X), X.p));
}

// ---------------------------------------------------------------------------
// jacobian
// ---------------------------------------------------------------------------

// H selects the d_p block: H = [0 0 I 0 0].
TEST(MekfPositionOutput, JacobianStructure) {
  Eigen::Matrix<double, 3, 15> H = PositionOutput::jacobian();
  Eigen::Matrix<double, 3, 15> expected = Eigen::Matrix<double, 3, 15>::Zero();
  expected.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  EXPECT(meq(H, expected));
}

// Jacobian is constant (no state dependence) — naive world-frame output.
TEST(MekfPositionOutput, JacobianIsConstant) {
  EXPECT(meq(PositionOutput::jacobian(), PositionOutput::jacobian()));
}

// Finite-difference check: H = d h(Retract(X, eps)) / d eps at eps = 0.
TEST(MekfPositionOutput, JacobianMatchesFiniteDifference) {
  const MekfState X = exampleState();
  const Eigen::Matrix<double, 3, 15> H = PositionOutput::jacobian();
  const Eigen::Vector3d h0 = PositionOutput::predict(X);
  constexpr double h = 1e-7;

  Eigen::Matrix<double, 3, 15> Hfd;
  for (int j = 0; j < 15; ++j) {
    Tangent e = Tangent::Zero();
    e(j) = h;
    const MekfState Xp = traits<MekfState>::Retract(X, e);
    Hfd.col(j) = (PositionOutput::predict(Xp) - h0) / h;
  }
  EXPECT(meq(H, Hfd, 1e-6));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
