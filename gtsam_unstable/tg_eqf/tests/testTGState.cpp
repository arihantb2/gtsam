/**
 * @file  testTGState.cpp
 * @brief Chart axioms for the State manifold.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/TGGeometry.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;

State makeXi() {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = Vector3(10.0, -5.0, 2.0);
  vp.col(1) = Vector3(1.0, 0.5, -0.2);
  State xi;
  xi.T = ExtendedPose3<2>(Rot3::Rz(0.5) * Rot3::Rx(0.3), vp);
  xi.b = (Vector9() << 0.01, -0.01, 0.02, -0.1, 0.0, 0.1, 0.05, 0.0, -0.05)
             .finished();
  xi.bm = Vector3(0.05, -0.02, 0.1);
  return xi;
}

using Tangent = State::TangentVector;

Tangent makeDelta() {
  Tangent delta;
  delta << 0.1, -0.2, 0.05, 1.0, 2.0, 3.0, -1.0, 0.5, 0.2, 0.01, 0.02, 0.03,
      -0.01, 0.0, 0.01, 0.0, -0.02, 0.01, 0.03, -0.04, 0.02;
  return delta;
}

/// Left-translate the navigation block by an SE_2(3) element, biases untouched.
State leftTranslate(const Rot3& R_s, const Vector3& v_s, const Vector3& p_s,
                    const State& xi) {
  Eigen::Matrix<double, 3, 2> vp_s;
  vp_s.col(0) = v_s;
  vp_s.col(1) = p_s;
  State out = xi;
  out.T = ExtendedPose3<2>(R_s, vp_s) * xi.T;
  return out;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace equality {

// Equals is the oracle the rest of the suite reads through, so check that it
// can fail as well as pass.
TEST(State, EqualsSeparatesStates) {
  const State a = fixture::makeXi();
  EXPECT(traits<State>::Equals(a, a, fixture::kTol));

  State b = a;
  Eigen::Matrix<double, 3, 2> vp_b;
  vp_b.col(0) = b.T.x(0) + Vector3(1.0, 0.0, 0.0);
  vp_b.col(1) = b.T.x(1);
  b.T = ExtendedPose3<2>(b.T.rotation(), vp_b);
  EXPECT(!traits<State>::Equals(a, b, fixture::kTol));

  State c = a;
  c.b(8) += 1.0;
  EXPECT(!traits<State>::Equals(a, c, fixture::kTol));

  State d = a;
  d.bm(2) += 1.0;
  EXPECT(!traits<State>::Equals(a, d, fixture::kTol));
}

}  // namespace equality
/* ************************************************************************* */

/* ************************************************************************* */
namespace chart {

// Retract(xi, 0) = xi and Local(xi, xi) = 0: the chart is centred where it
// says.
TEST(State, ChartIsCentredAtItsBasePoint) {
  const State xi = fixture::makeXi();
  const fixture::Tangent zero = fixture::Tangent::Zero();

  EXPECT(traits<State>::Equals(xi, traits<State>::Retract(xi, zero),
                               fixture::kTol));
  EXPECT(assert_equal((Vector)zero, (Vector)traits<State>::Local(xi, xi),
                      fixture::kTol));
}

// Local inverts Retract, at the manifold identity and away from it. The
// SE_2(3) logarithm makes these two genuinely different code paths.
TEST(State, RetractLocalRoundTrip) {
  const fixture::Tangent delta = fixture::makeDelta();

  for (const State& xi_ref : {State::identity(), fixture::makeXi()}) {
    const State xi = traits<State>::Retract(xi_ref, delta);
    EXPECT(assert_equal((Vector)delta, (Vector)traits<State>::Local(xi_ref, xi),
                        1e-7));
  }
}

// The defining property of the EqF chart: its coordinates are SE_2(3) logarithm
// coordinates, so they are invariant under a left translation of the extended
// pose. Coordinates that took plain differences on v and p would not be.
TEST(State, ChartIsLeftInvariant) {
  const State xi_ref = fixture::makeXi();
  const State xi = traits<State>::Retract(xi_ref, fixture::makeDelta());

  const Rot3 R_s = Rot3::Ry(0.9) * Rot3::Rz(-0.4);
  const Vector3 v_s(-3.0, 1.0, 0.7);
  const Vector3 p_s(2.0, -0.5, 4.0);

  EXPECT(assert_equal((Vector)traits<State>::Local(xi_ref, xi),
                      (Vector)traits<State>::Local(
                          fixture::leftTranslate(R_s, v_s, p_s, xi_ref),
                          fixture::leftTranslate(R_s, v_s, p_s, xi)),
                      1e-9));
}

// At zero navigation-block displacement, the bias block is exactly linear in
// the raw bias difference, but negated.
TEST(State, BiasBlockIsLinearAtZeroNavigationDisplacement) {
  const State xi_ref = fixture::makeXi();
  State xi = xi_ref;
  xi.b.segment<3>(3) += Vector3(0.3, -0.2, 0.1);

  fixture::Tangent expected = fixture::Tangent::Zero();
  expected.segment<3>(12) = -Vector3(0.3, -0.2, 0.1);

  EXPECT(assert_equal((Vector)expected,
                      (Vector)traits<State>::Local(xi_ref, xi), fixture::kTol));
}

// The same holds for the magnetometer-bias block, which the semidirect
// exponential moves by -eps_bm when the attitude is not displaced.
TEST(State, MagBiasBlockIsLinearAtZeroNavigationDisplacement) {
  const State xi_ref = fixture::makeXi();
  State xi = xi_ref;
  xi.bm += Vector3(0.2, 0.1, -0.3);

  fixture::Tangent expected = fixture::Tangent::Zero();
  expected.segment<3>(18) = -Vector3(0.2, 0.1, -0.3);

  EXPECT(assert_equal((Vector)expected,
                      (Vector)traits<State>::Local(xi_ref, xi), fixture::kTol));
}

}  // namespace chart
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
