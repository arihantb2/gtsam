/**
 * @file  testTGVirtualBiasOutput.cpp
 * @brief The b_v = 0 pseudo-measurement that pins the unobservable virtual
 * bias.
 *
 * The state action is affine on the bias block, so this output matrix carries
 * no linearization error at all. Exactness is the only property worth
 * asserting: a linear map that reproduces the residual on a spanning set of
 * errors is the only such map, so exactness already determines C* completely.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

using Tangent = Eigen::Matrix<double, 18, 1>;

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

/// Virtual bias the true state at eps carries, relative to the estimate's.
Eigen::Vector3d residual(const State& xi_ref, const TGElement& g,
                         const Tangent& eps) {
  return phi(g, traits<State>::Retract(xi_ref, eps)).b_v - phi(g, xi_ref).b_v;
}

Tangent makeEps() {
  Tangent eps;
  eps << 0.4, -0.3, 0.5, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.2, -0.15, 0.1, -0.2,
      0.3, 0.05, 0.1, -0.25, 0.15;
  return eps;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_matrix {

// C* reproduces the residual exactly, at an error far larger than any the
// filter sees. Checked at the group identity and away from it.
TEST(VirtualBiasOutput, CstarIsExact) {
  const State xi_ref = fixture::makeXi();
  const fixture::Tangent eps = fixture::makeEps();

  for (const TGElement& g : {TGElement::Identity(), fixture::makeX()}) {
    EXPECT(assert_equal(
        (Vector)fixture::residual(xi_ref, g, eps),
        (Vector)(VirtualBiasMeasurement::jacobian_Cstar(g) * eps), 1e-12));
  }
}

// jacobian_Cstar takes only the group element, because the bias coordinates are
// Euclidean: the reference biases cancel out of the residual, so the same
// matrix is exact at any reference state.
TEST(VirtualBiasOutput, CstarDoesNotDependOnTheReferenceState) {
  const TGElement g = fixture::makeX();
  const fixture::Tangent eps = fixture::makeEps();
  const Eigen::Vector3d predicted =
      VirtualBiasMeasurement::jacobian_Cstar(g) * eps;

  for (const State& xi_ref : {State::identity(), fixture::makeXi()}) {
    EXPECT(assert_equal((Vector)fixture::residual(xi_ref, g, eps),
                        (Vector)predicted, 1e-12));
  }
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
