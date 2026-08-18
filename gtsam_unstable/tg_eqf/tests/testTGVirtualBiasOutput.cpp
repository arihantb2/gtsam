/**
 * @file  testTGVirtualBiasOutput.cpp
 * @brief Unit tests for the b_v = 0 virtual-bias pseudo-measurement (Eq. B.20).
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>
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

/* ************************************************************************* */
namespace output_matrix {

using Tangent = Eigen::Matrix<double, 18, 1>;

TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

/// Residual the filter linearizes: the virtual bias the true state at eps
/// carries, relative to the one the estimate reports.
Eigen::Vector3d residual(const State& xi_ref, const TGElement& g,
                         const Tangent& eps) {
  return phi(g, traits<State>::Retract(xi_ref, eps)).b_v - phi(g, xi_ref).b_v;
}

// With g = identity the action is trivial and C* selects b_v outright.
TEST(VirtualBiasOutput, CstarSelectsVirtualBiasAtIdentityGroup) {
  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  EXPECT(meq(expected,
             VirtualBiasMeasurement::jacobian_Cstar(TGElement::Identity())));
}

// C* is the b_v row of the inverse adjoint the state action applies to biases.
TEST(VirtualBiasOutput, CstarIsTheVirtualBiasRowOfTheInverseAdjoint) {
  const TGElement g = makeX();
  const Eigen::Matrix<double, 9, 9> Ad_inv = detail::Ad_SE23_inv(g);

  Eigen::Matrix<double, 3, 18> expected = Eigen::Matrix<double, 3, 18>::Zero();
  expected.block<3, 9>(0, 9) = Ad_inv.block<3, 9>(6, 0);

  EXPECT(meq(expected, VirtualBiasMeasurement::jacobian_Cstar(g)));
}

// The bias block of the state action is affine in the error, so C* carries no
// linearization error at all: it reproduces the residual exactly, at an error
// far larger than any the filter sees.
TEST(VirtualBiasOutput, CstarIsExactAtFiniteError) {
  const State xi_ref = makeXi();
  const TGElement g = makeX();
  const Eigen::Matrix<double, 3, 18> C =
      VirtualBiasMeasurement::jacobian_Cstar(g);

  Tangent eps;
  eps << 0.4, -0.3, 0.5, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.2, -0.15, 0.1, -0.2,
      0.3, 0.05, 0.1, -0.25, 0.15;

  EXPECT(veq(residual(xi_ref, g, eps), C * eps, 1e-12));
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
