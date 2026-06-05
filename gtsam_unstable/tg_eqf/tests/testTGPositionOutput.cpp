#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
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
    const TGState& xi, const Eigen::Vector3d& pi, double h = 1e-6) {
  Eigen::Matrix<double, 3, 18> J;
  const Eigen::Vector3d f0 = PositionMeasurement::predict(xi, pi);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xip = traits<TGState>::Retract(xi, e);
    J.col(j) = (PositionMeasurement::predict(xip, pi) - f0) / h;
  }
  return J;
}

static bool checkEquivariance(const TGGroupElement& X, const TGState& xi,
                              const Eigen::Vector3d& pi) {
  const Eigen::Vector3d y = PositionMeasurement::predict(xi, pi);
  const Eigen::Vector3d lhs =
      PositionMeasurement::predict(phi(X, xi), pi);
  const Eigen::Vector3d rhs =
      PositionMeasurement::output_action(X, y);
  return veq(lhs, rhs, kTolL);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(PositionOutput, PredictAtIdentityIsZeroWhenPiEqualsP) {
  TGState xi = TGState::identity();
  const Eigen::Vector3d pi(1.0, -2.0, 0.5);
  xi.p = pi;
  EXPECT(veq(Eigen::Vector3d::Zero(),
             PositionMeasurement::predict(xi, pi)));
}

TEST(PositionOutput, PredictMatchesBodyFrameResidual) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Vector3d expected = xi.R.unrotate(pi - xi.p);
  EXPECT(veq(expected, PositionMeasurement::predict(xi, pi)));
}

// ---------------------------------------------------------------------------
// output_action
// ---------------------------------------------------------------------------

TEST(PositionOutput, OutputActionAtIdentityIsNoop) {
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  EXPECT(veq(y, PositionMeasurement::output_action(
                  TGGroupElement::Identity(), y)));
}

TEST(PositionOutput, OutputActionMatchesFormula) {
  const TGGroupElement X = makeX();
  const Eigen::Vector3d y(0.3, -0.1, 0.2);
  const Eigen::Vector3d expected = X.R_X.unrotate(y - X.p_X);
  EXPECT(veq(expected, PositionMeasurement::output_action(X, y)));
}

// ---------------------------------------------------------------------------
// equivariance
// ---------------------------------------------------------------------------

TEST(PositionOutput, EquivarianceAtIdentity) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  EXPECT(checkEquivariance(TGGroupElement::Identity(), xi, pi));
}

TEST(PositionOutput, EquivarianceForGeneralX) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  EXPECT(checkEquivariance(makeX(), xi, pi));
}

TEST(PositionOutput, EquivarianceForExpmapElements) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4,
         0.1, -0.1, 0.05, -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkEquivariance(TGGroupElement::Expmap(logX), xi, pi));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2,
         0.05, 0.05, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkEquivariance(TGGroupElement::Expmap(logY), xi, pi));
}

// ---------------------------------------------------------------------------
// innovation
// ---------------------------------------------------------------------------

TEST(PositionOutput, InnovationIsNegativePredict) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Vector3d expected =
      -PositionMeasurement::predict(xi, pi);
  EXPECT(veq(expected, PositionMeasurement::innovation(pi, xi)));
}

TEST(PositionOutput, InnovationZeroWhenPiEqualsP) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi = xi.p;
  EXPECT(veq(Eigen::Vector3d::Zero(),
             PositionMeasurement::innovation(pi, xi)));
}

// ---------------------------------------------------------------------------
// Jacobians
// ---------------------------------------------------------------------------

TEST(PositionOutput, JacobianC0AtIdentity) {
  const TGState xi_ref = TGState::identity();
  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, PositionMeasurement::jacobian_C0(xi_ref)));
}

TEST(PositionOutput, JacobianC0PositionBlockMatchesNumerical) {
  const TGState xi_ref = TGState::identity();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Matrix<double, 3, 18> H_num =
      numericalJacobian(xi_ref, pi);
  const Eigen::Matrix<double, 3, 18> H_anal =
      PositionMeasurement::jacobian_C0(xi_ref);

  // C0 is first-order at the fixed origin: rotation block is zero by design.
  EXPECT(meq(H_anal.block<3, 3>(0, 0),
             Eigen::Matrix3d::Zero(), kTol));
  EXPECT(meq(H_anal.block<3, 3>(0, 3), H_num.block<3, 3>(0, 3), 1e-5));
}

TEST(PositionOutput, JacobianCstarMatchesDocumentedFormula) {
  const TGState xi = makeXi();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  const Eigen::Vector3d y = PositionMeasurement::predict(xi, pi);

  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) =
      0.5 * gtsam::skewSymmetric(y + xi.p);
  expected.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

  EXPECT(meq(expected, PositionMeasurement::jacobian_Cstar(xi, pi)));
}

TEST(PositionOutput, JacobianCstarAtIdentityWithPi) {
  const TGState xi = TGState::identity();
  const Eigen::Vector3d pi(2.5, -1.0, 1.2);
  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 0) =
      0.5 * gtsam::skewSymmetric(pi);
  expected.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, PositionMeasurement::jacobian_Cstar(xi, pi)));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
