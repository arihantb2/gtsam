/**
 * @file  testTFGMagnetometerOutput.cpp
 * @brief Unit tests for the biased magnetometer output h = R^T (m - gamma_m),
 *        its output action, and its output matrix C0.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tfg_inekf/MagnetometerOutput.h>
#include <gtsam_unstable/tfg_inekf/Symmetry.h>

using namespace tfg;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;

/// Reference field in the global frame (normalized units).
const Eigen::Vector3d kField(0.4, 0.05, 0.9);

bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
         double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}
bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
         double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

TwoFrameGroup makeState() {
  return TwoFrameGroup::FromState(
      Rot3::Rz(0.4) * Rot3::Rx(0.2), Eigen::Vector3d(0.3, 0.1, -0.4),
      Eigen::Vector3d(1.0, -2.0, 0.5), Eigen::Vector3d(0.02, -0.01, 0.03),
      Eigen::Vector3d(-0.04, 0.05, 0.0), Eigen::Vector3d(0.08, -0.05, 0.03));
}

TwoFrameGroup makeX() {
  return TwoFrameGroup(
      Rot3::Ry(0.3) * Rot3::Rz(-0.1), Eigen::Vector3d(0.2, -0.3, 0.1),
      Eigen::Vector3d(-0.5, 1.0, 2.0), Eigen::Vector3d(0.05, 0.0, -0.05),
      Eigen::Vector3d(0.1, -0.1, 0.2), Eigen::Vector3d(-0.1, 0.2, 0.05));
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_action {

// h(xi) = R^T (m - gamma_m) is the biased body-frame reading R^T m + b_m.
TEST(MagnetometerOutput, PredictIsTheBiasedBodyField) {
  const TwoFrameGroup xi = fixture::makeState();
  EXPECT(fixture::veq(xi.R.unrotate(fixture::kField) + xi.bias_mag(),
                      MagnetometerOutput::predict(xi, fixture::kField),
                      fixture::kTolL));
}

// rho is a right action on the output space: rho_I = id and
// rho_{XY} = rho_Y o rho_X.
TEST(MagnetometerOutput, OutputActionIsARightAction) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const TwoFrameGroup X = fixture::makeX(), Y = fixture::makeState();

  EXPECT(fixture::veq(
      y, MagnetometerOutput::output_action(TwoFrameGroup::Identity(), y)));
  EXPECT(fixture::veq(MagnetometerOutput::output_action(X * Y, y),
                      MagnetometerOutput::output_action(
                          Y, MagnetometerOutput::output_action(X, y)),
                      fixture::kTolL));
}

// Output equivariance h(phi(X, xi)) = rho_X(h(xi)), including elements that
// move the magnetometer bias.
TEST(MagnetometerOutput, OutputEquivariance) {
  const TwoFrameGroup xi = fixture::makeState();
  const TwoFrameGroup X = fixture::makeX();

  EXPECT(fixture::veq(MagnetometerOutput::predict(phi(X, xi), fixture::kField),
                      MagnetometerOutput::output_action(
                          X, MagnetometerOutput::predict(xi, fixture::kField)),
                      fixture::kTolL));
}

}  // namespace output_action
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_matrix {

// C0 = [ y_hat^ 0 0 0 0 -I ].
TEST(MagnetometerOutput, C0Structure) {
  const TwoFrameGroup xi = fixture::makeState();
  const Eigen::Vector3d y_hat =
      MagnetometerOutput::predict(xi, fixture::kField);
  const Eigen::Matrix<double, 3, 18> C =
      MagnetometerOutput::jacobian(xi, fixture::kField);

  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = skewSymmetric(y_hat);
  expected.block<3, 3>(0, 15) = -Eigen::Matrix3d::Identity();
  EXPECT(fixture::meq(C, expected));
}

// C0 is the derivative of h(xi_hat * Expmap(eps)) at eps = 0, the chart used
// by ManifoldEKF::update.
TEST(MagnetometerOutput, C0MatchesFiniteDifference) {
  const TwoFrameGroup xi = fixture::makeState();
  const Eigen::Matrix<double, 3, 18> C_num =
      numericalDerivative11<Eigen::Vector3d, TwoFrameGroup>(
          [](const TwoFrameGroup& x) {
            return MagnetometerOutput::predict(x, fixture::kField);
          },
          xi);

  EXPECT(fixture::meq(MagnetometerOutput::jacobian(xi, fixture::kField), C_num,
                      1e-6));
}

// The attitude block y_hat^ annihilates the predicted body-frame field: the
// magnetometer cannot see rotation about m.
TEST(MagnetometerOutput, AttitudeBlockIsBlindToRotationAboutTheField) {
  const TwoFrameGroup xi = fixture::makeState();
  const Eigen::Vector3d y_hat =
      MagnetometerOutput::predict(xi, fixture::kField);
  const Eigen::Matrix<double, 3, 18> C =
      MagnetometerOutput::jacobian(xi, fixture::kField);

  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), C.leftCols<3>() * y_hat));
  EXPECT((C.leftCols<3>() * xi.R.unrotate(Eigen::Vector3d::UnitZ())).norm() >
         0.1);
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
