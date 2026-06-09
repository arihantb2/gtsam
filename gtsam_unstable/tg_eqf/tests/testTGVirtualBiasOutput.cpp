#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

using namespace tgeqf;
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

static TGState makeXi() {
  TGState xi;
  xi.R   = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.v   = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.p   = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_a = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_v = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

TEST(VirtualBiasOutput, PredictReturnsVirtualBias) {
  const TGState xi = makeXi();
  EXPECT(veq(xi.b_v, VirtualBiasMeasurement::predict(xi)));
}

TEST(VirtualBiasOutput, PredictZeroAtIdentity) {
  EXPECT(veq(Eigen::Vector3d::Zero(),
             VirtualBiasMeasurement::predict(TGState::identity())));
}

TEST(VirtualBiasOutput, InnovationIsNegativePredict) {
  const TGState xi = makeXi();
  EXPECT(veq(-xi.b_v, VirtualBiasMeasurement::innovation(xi)));
}

// ---------------------------------------------------------------------------
// Jacobian C0 (origin chart; realises the Eq. B.20 constraint h = b_v = 0)
// ---------------------------------------------------------------------------

// Independent finite-difference in the State::Retract chart (the chart the
// EquivariantFilter linearises measurements in): C0 = d(predict)/d(eps).
static Eigen::Matrix<double, 3, 18> numericalC0(const TGState& xi,
                                                double h = 1e-6) {
  Eigen::Matrix<double, 3, 18> J;
  const Eigen::Vector3d f0 = VirtualBiasMeasurement::predict(xi);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xip = traits<TGState>::Retract(xi, e);
    J.col(j) = (VirtualBiasMeasurement::predict(xip) - f0) / h;
  }
  return J;
}

TEST(VirtualBiasOutput, JacobianC0IsIdentityInVirtualBiasBlock) {
  Eigen::Matrix<double, 3, 18> expected =
      Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  EXPECT(meq(expected, VirtualBiasMeasurement::jacobian_C0(makeXi())));
  EXPECT(meq(expected,
             VirtualBiasMeasurement::jacobian_C0(TGState::identity())));
}

TEST(VirtualBiasOutput, JacobianC0MatchesNumerical) {
  const TGState xi = makeXi();
  EXPECT(meq(VirtualBiasMeasurement::jacobian_C0(xi), numericalC0(xi), 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
