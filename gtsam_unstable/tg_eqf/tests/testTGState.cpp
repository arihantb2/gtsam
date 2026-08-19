/**
 * @file  testTGState.cpp
 * @brief Chart axioms for the State manifold.
 *
 * State carries the EqF chart: the SE_2(3) logarithm on the navigation block
 * and the identity on the biases. Everything above this file -- the symmetry
 * Jacobians, the lift differential, the output matrices, the reset -- is
 * differentiated in this chart, so the axioms here are the base of the suite.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/State.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;

State makeXi() {
  State xi;
  xi.R = Rot3::Rz(0.5) * Rot3::Rx(0.3);
  xi.v = Eigen::Vector3d(10.0, -5.0, 2.0);
  xi.p = Eigen::Vector3d(1.0, 0.5, -0.2);
  xi.b_w = Eigen::Vector3d(0.01, -0.01, 0.02);
  xi.b_a = Eigen::Vector3d(-0.1, 0.0, 0.1);
  xi.b_v = Eigen::Vector3d(0.05, 0.0, -0.05);
  return xi;
}

Eigen::Matrix<double, 18, 1> makeDelta() {
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.1, -0.2, 0.05, 1.0, 2.0, 3.0, -1.0, 0.5, 0.2, 0.01, 0.02, 0.03,
      -0.01, 0.0, 0.01, 0.0, -0.02, 0.01;
  return delta;
}

/// Left-translate the navigation block by an SE_2(3) element, biases untouched.
State leftTranslate(const Rot3& R_s, const Eigen::Vector3d& v_s,
                    const Eigen::Vector3d& p_s, const State& xi) {
  State out = xi;
  out.R = R_s * xi.R;
  out.v = R_s.rotate(xi.v) + v_s;
  out.p = R_s.rotate(xi.p) + p_s;
  return out;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace equality {

// Equals is the oracle every other test in the suite reads through, so pin that
// it can actually fail. A vacuous Equals would silently pass everything.
TEST(State, EqualsSeparatesStates) {
  const State a = fixture::makeXi();
  EXPECT(traits<State>::Equals(a, a, fixture::kTol));

  State b = a;
  b.v.x() += 1.0;
  EXPECT(!traits<State>::Equals(a, b, fixture::kTol));

  State c = a;
  c.b_v.z() += 1.0;
  EXPECT(!traits<State>::Equals(a, c, fixture::kTol));
}

}  // namespace equality
/* ************************************************************************* */

/* ************************************************************************* */
namespace chart {

// Retract(xi, 0) = xi and Local(xi, xi) = 0: the chart is centred where it says.
TEST(State, ChartIsCentredAtItsBasePoint) {
  const State xi = fixture::makeXi();
  const Eigen::Matrix<double, 18, 1> zero =
      Eigen::Matrix<double, 18, 1>::Zero();

  EXPECT(traits<State>::Equals(xi, traits<State>::Retract(xi, zero),
                               fixture::kTol));
  EXPECT(assert_equal((Vector)zero, (Vector)traits<State>::Local(xi, xi),
                      fixture::kTol));
}

// Local inverts Retract, at the manifold identity and away from it. The
// SE_2(3) logarithm makes these two genuinely different code paths.
TEST(State, RetractLocalRoundTrip) {
  const Eigen::Matrix<double, 18, 1> delta = fixture::makeDelta();

  for (const State& xi_ref : {State::identity(), fixture::makeXi()}) {
    const State xi = traits<State>::Retract(xi_ref, delta);
    EXPECT(assert_equal((Vector)delta, (Vector)traits<State>::Local(xi_ref, xi),
                        1e-7));
  }
}

// The defining property of the EqF chart: its coordinates are SE_2(3) logarithm
// coordinates, so they are invariant under a left translation of the extended
// pose. A naive product chart (Logmap on R, differences on v and p) fails this
// as soon as the translating rotation is non-trivial.
TEST(State, ChartIsLeftInvariant) {
  const State xi_ref = fixture::makeXi();
  const State xi = traits<State>::Retract(xi_ref, fixture::makeDelta());

  const Rot3 R_s = Rot3::Ry(0.9) * Rot3::Rz(-0.4);
  const Eigen::Vector3d v_s(-3.0, 1.0, 0.7);
  const Eigen::Vector3d p_s(2.0, -0.5, 4.0);

  EXPECT(assert_equal(
      (Vector)traits<State>::Local(xi_ref, xi),
      (Vector)traits<State>::Local(fixture::leftTranslate(R_s, v_s, p_s, xi_ref),
                                   fixture::leftTranslate(R_s, v_s, p_s, xi)),
      1e-9));
}

// The bias block is Euclidean and uncoupled from the navigation block: shifting
// a bias moves only its own coordinates.
TEST(State, BiasBlockIsEuclidean) {
  const State xi_ref = fixture::makeXi();
  State xi = xi_ref;
  xi.b_a += Eigen::Vector3d(0.3, -0.2, 0.1);

  Eigen::Matrix<double, 18, 1> expected =
      Eigen::Matrix<double, 18, 1>::Zero();
  expected.segment<3>(12) = Eigen::Vector3d(0.3, -0.2, 0.1);

  EXPECT(assert_equal((Vector)expected, (Vector)traits<State>::Local(xi_ref, xi),
                      fixture::kTol));
}

}  // namespace chart
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
