/**
 * @file  testTGVirtualBiasOutput.cpp
 * @brief The b_v = 0 pseudo-measurement that pins the unobservable virtual
 * bias.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

using Tangent = Eigen::Matrix<double, 18, 1>;
constexpr double kTolL = 1e-6;

State makeXi() {
  State xi;
  xi.R = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_a = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_v = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_matrix {

// jacobian_Cstar is the chart derivative of phi(g, Retract(xi_ref, .)).b_v at
// eps = 0.
TEST(VirtualBiasOutput, CstarMatchesNumerical) {
  const State xi_ref = fixture::makeXi();

  for (const TGElement& g : {TGElement::Identity(), fixture::makeX()}) {
    const Eigen::Matrix<double, 3, 18> H_num =
        numericalDerivative11<Eigen::Vector3d, State>(
            [&](const State& x) { return phi(g, x).b_v; }, xi_ref);

    EXPECT(assert_equal((Matrix)VirtualBiasMeasurement::jacobian_Cstar(xi_ref, g),
                        (Matrix)H_num, fixture::kTolL));
  }
}

// jacobian_Cstar depends on xi_ref (unlike DVL/position): the chart couples
// navigation into the bias value whenever the reference bias is non-zero.
TEST(VirtualBiasOutput, CstarDependsOnTheReferenceBias) {
  State xi_ref_zero_bias = fixture::makeXi();
  xi_ref_zero_bias.b_w.setZero();
  xi_ref_zero_bias.b_a.setZero();
  xi_ref_zero_bias.b_v.setZero();
  const State xi_ref = fixture::makeXi();
  const TGElement g = fixture::makeX();

  const Eigen::Matrix<double, 3, 18> Cstar_zero_bias =
      VirtualBiasMeasurement::jacobian_Cstar(xi_ref_zero_bias, g);
  const Eigen::Matrix<double, 3, 18> Cstar =
      VirtualBiasMeasurement::jacobian_Cstar(xi_ref, g);

  EXPECT((Cstar - Cstar_zero_bias).norm() > 1e-3);
  // Both still agree on the bias columns, which do not depend on xi_ref.b.
  EXPECT(assert_equal((Matrix)Cstar_zero_bias.rightCols<9>(),
                      (Matrix)Cstar.rightCols<9>(), 1e-12));
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
