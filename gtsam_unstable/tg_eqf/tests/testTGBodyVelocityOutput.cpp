#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol  = 1e-9;
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

static TGState makeXi() {
  TGState xi;
  xi.R       = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.p       = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.v       = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_omega = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_v     = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_a     = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

static TGGroupElement makeX() {
  TGGroupElement X;
  X.R_X = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.p_X = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.v_X = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a   = {Eigen::Vector3d(0.1, -0.1, 0.05),
            Eigen::Vector3d(-0.2, 0.3, 0.0),
            Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

static Eigen::Matrix<double, 3, 18> numericalJacobian(
    const TGState& xi, double h = 1e-6) {
  Eigen::Matrix<double, 3, 18> J;
  const Eigen::Vector3d f0 = DVLMeasurement::predict(xi);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xip = traits<TGState>::Retract(xi, e);
    J.col(j) = (DVLMeasurement::predict(xip) - f0) / h;
  }
  return J;
}

static bool checkEquivariance(const TGGroupElement& X, const TGState& xi) {
  const Eigen::Vector3d y = DVLMeasurement::predict(xi);
  const Eigen::Vector3d lhs =
      DVLMeasurement::predict(phi(X, xi));
  const Eigen::Vector3d rhs =
      DVLMeasurement::output_action(X, y);
  return veq(lhs, rhs, kTolL);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, PredictAtIdentityIsZero) {
  const TGState xi = TGState::identity();
  EXPECT(veq(Eigen::Vector3d::Zero(), DVLMeasurement::predict(xi)));
}

TEST(BodyVelocityOutput, PredictMatchesBodyFrameVelocity) {
  const TGState xi = makeXi();
  const Eigen::Vector3d expected = xi.R.unrotate(xi.v);
  EXPECT(veq(expected, DVLMeasurement::predict(xi)));
}

// ---------------------------------------------------------------------------
// output_action
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, OutputActionAtIdentityIsNoop) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  EXPECT(veq(y, DVLMeasurement::output_action(
                  TGGroupElement::Identity(), y)));
}

TEST(BodyVelocityOutput, OutputActionMatchesFormula) {
  const TGGroupElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d expected =
      X.R_X.unrotate(y) + X.R_X.unrotate(X.v_X);
  EXPECT(veq(expected, DVLMeasurement::output_action(X, y)));
}

// ---------------------------------------------------------------------------
// equivariance
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, EquivarianceAtIdentity) {
  const TGState xi = makeXi();
  EXPECT(checkEquivariance(TGGroupElement::Identity(), xi));
}

TEST(BodyVelocityOutput, EquivarianceForGeneralX) {
  const TGState xi = makeXi();
  EXPECT(checkEquivariance(makeX(), xi));
}

TEST(BodyVelocityOutput, EquivarianceForExpmapElements) {
  const TGState xi = makeXi();

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4,
         0.1, -0.1, 0.05, -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkEquivariance(TGGroupElement::Expmap(logX), xi));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2,
         0.05, 0.05, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkEquivariance(TGGroupElement::Expmap(logY), xi));
}

// ---------------------------------------------------------------------------
// innovation
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, InnovationIsZMinusPredict) {
  const TGState xi = makeXi();
  const Eigen::Vector3d z(0.2, -0.3, 0.1);
  const Eigen::Vector3d expected =
      z - DVLMeasurement::predict(xi);
  EXPECT(veq(expected, DVLMeasurement::innovation(z, xi)));
}

TEST(BodyVelocityOutput, InnovationZeroWhenZMatchesPredict) {
  const TGState xi = makeXi();
  const Eigen::Vector3d z = DVLMeasurement::predict(xi);
  EXPECT(veq(Eigen::Vector3d::Zero(),
             DVLMeasurement::innovation(z, xi)));
}

// ---------------------------------------------------------------------------
// Jacobians
// ---------------------------------------------------------------------------

TEST(BodyVelocityOutput, JacobianAtIdentity) {
  const TGState xi_ref = TGState::identity();
  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, DVLMeasurement::jacobian(xi_ref)));
}

TEST(BodyVelocityOutput, JacobianMatchesDocumentedFormula) {
  const TGState xi = makeXi();
  const Eigen::Vector3d body_v = DVLMeasurement::predict(xi);

  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) = gtsam::skewSymmetric(body_v);
  expected.block<3, 3>(0, 6) = xi.R.matrix().transpose();

  EXPECT(meq(expected, DVLMeasurement::jacobian(xi)));
}

TEST(BodyVelocityOutput, JacobianMatchesNumerical) {
  const TGState xi = makeXi();
  const Eigen::Matrix<double, 3, 18> H_num = numericalJacobian(xi);
  const Eigen::Matrix<double, 3, 18> H_anal = DVLMeasurement::jacobian(xi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

TEST(BodyVelocityOutput, JacobianRotationBlockZeroAtIdentity) {
  const TGState xi_ref = TGState::identity();
  const Eigen::Matrix<double, 3, 18> H =
      DVLMeasurement::jacobian(xi_ref);
  EXPECT(meq(H.block<3, 3>(0, 0), Eigen::Matrix3d::Zero(), kTol));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
