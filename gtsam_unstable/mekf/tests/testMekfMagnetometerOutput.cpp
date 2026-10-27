/**
 * @file  testMekfMagnetometerOutput.cpp
 * @brief Unit tests for the biased magnetometer output (h = R^T m + b_mag, H,
 *        FD check).
 *
 * Tangent block order: see ../README.md. Magnetometer bias at indices 15..17.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/mekf/MagnetometerOutput.h>

using namespace mekf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;

/// Reference field in the navigation frame (normalized units).
const Eigen::Vector3d kField(0.4, 0.05, 0.9);

bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
         double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}
bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
         double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

MekfState exampleState() {
  return MekfState(
      Rot3::Rz(0.4) * Rot3::Rx(0.2), Eigen::Vector3d(1.0, -0.5, 0.3),
      Eigen::Vector3d(2.0, 1.0, -1.0), Eigen::Vector3d(0.01, -0.02, 0.03),
      Eigen::Vector3d(-0.05, 0.1, 0.0), Eigen::Vector3d(0.05, -0.08, 0.03));
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace predict {

// h(xi) = R^T m + b_mag: the reference field seen in the body frame, offset by
// the hard-iron bias.
TEST(MekfMagnetometerOutput, PredictReturnsBiasedBodyField) {
  const MekfState X = fixture::exampleState();
  EXPECT(fixture::veq(X.R.unrotate(fixture::kField) + X.b_mag,
                      MagnetometerOutput::predict(X, fixture::kField)));
}

// At the identity attitude with no bias the reading is the reference field.
TEST(MekfMagnetometerOutput, PredictAtIdentityIsTheField) {
  EXPECT(fixture::veq(
      fixture::kField,
      MagnetometerOutput::predict(MekfState::identity(), fixture::kField)));
}

}  // namespace predict
/* ************************************************************************* */

/* ************************************************************************* */
namespace jacobian {

// H = [ [R^T m]^x | 0 | 0 | 0 | 0 | I ]; the other blocks vanish.
TEST(MekfMagnetometerOutput, JacobianStructure) {
  const MekfState X = fixture::exampleState();
  const Eigen::Matrix<double, 3, 18> H =
      MagnetometerOutput::jacobian(X, fixture::kField);

  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = skewSymmetric(X.R.unrotate(fixture::kField));
  expected.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  EXPECT(fixture::meq(H, expected));
}

// Finite-difference check: H = d h(Retract(X, eps)) / d eps at eps = 0.
TEST(MekfMagnetometerOutput, JacobianMatchesFiniteDifference) {
  const MekfState X = fixture::exampleState();
  const Eigen::Matrix<double, 3, 18> H =
      MagnetometerOutput::jacobian(X, fixture::kField);

  const Eigen::Matrix<double, 3, 18> Hfd =
      numericalDerivative11<Eigen::Vector3d, MekfState>(
          [](const MekfState& x) {
            return MagnetometerOutput::predict(x, fixture::kField);
          },
          X);
  EXPECT(fixture::meq(H, Hfd, 1e-6));
}

// The attitude block [R^T m]^x annihilates the body-frame field direction:
// the magnetometer cannot see rotation about m.
TEST(MekfMagnetometerOutput, AttitudeBlockIsBlindToRotationAboutTheField) {
  const MekfState X = fixture::exampleState();
  const Eigen::Matrix<double, 3, 18> H =
      MagnetometerOutput::jacobian(X, fixture::kField);
  const Eigen::Vector3d field_body = X.R.unrotate(fixture::kField);

  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), H.leftCols<3>() * field_body));
  EXPECT((H.leftCols<3>() * X.R.unrotate(Eigen::Vector3d::UnitZ())).norm() >
         0.1);
}

}  // namespace jacobian
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
