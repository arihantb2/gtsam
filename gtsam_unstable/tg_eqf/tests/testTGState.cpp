/**
 * @file  testState.cpp
 * @brief Unit tests for the State manifold (identity, T_matrix,
 * bias_vector, Retract/Local).

 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/State.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-9;

// assert_equal overloads take dynamic gtsam::Vector / gtsam::Matrix;
// explicit cast resolves ambiguity for fixed-size Eigen types.
static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

static bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

// ---------------------------------------------------------------------------
// State::identity
// ---------------------------------------------------------------------------

TEST(State, IdentityIsZero) {
  State xi = State::identity();
  EXPECT(xi.R.equals(Rot3::Identity(), kTol));
  EXPECT(veq(xi.v, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.p, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_w, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_a, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_v, Eigen::Vector3d::Zero()));
}

// Guards the struct's own default member initializers, independent of
// identity()'s explicit zeroing.
TEST(State, DefaultConstructedIsIdentity) {
  State xi;
  EXPECT(traits<State>::Equals(State::identity(), xi, kTol));
}

// ---------------------------------------------------------------------------
// State::T_matrix
// ---------------------------------------------------------------------------

TEST(State, TMatrixAtIdentityIsI5) {
  auto T = State::identity().T_matrix();
  EXPECT(meq(T, Eigen::Matrix<double, 5, 5>::Identity()));
}

TEST(State, TMatrixBlocks) {
  State xi;
  xi.R = Rot3::Rx(M_PI / 2.0);
  xi.v = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.p = Eigen::Vector3d(4.0, 5.0, 6.0);

  auto T = xi.T_matrix();

  EXPECT(meq(xi.R.matrix(), T.block<3, 3>(0, 0)));
  EXPECT(veq(xi.v, T.block<3, 1>(0, 3)));
  EXPECT(veq(xi.p, T.block<3, 1>(0, 4)));
  DOUBLES_EQUAL(1.0, T(3, 3), kTol);
  DOUBLES_EQUAL(1.0, T(4, 4), kTol);
  DOUBLES_EQUAL(0.0, T(3, 4), kTol);
  DOUBLES_EQUAL(0.0, T(4, 3), kTol);
  // bottom-left 2x3 block is zero
  EXPECT(meq(T.block<2, 3>(3, 0), Eigen::Matrix<double, 2, 3>::Zero()));
}

// ---------------------------------------------------------------------------
// State::bias_vector
// ---------------------------------------------------------------------------

TEST(State, BiasVectorPacking) {
  State xi = State::identity();
  xi.b_w = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.b_a = Eigen::Vector3d(4.0, 5.0, 6.0);
  xi.b_v = Eigen::Vector3d(7.0, 8.0, 9.0);

  auto b = xi.bias_vector();

  EXPECT(veq(xi.b_w, b.segment<3>(0)));
  EXPECT(veq(xi.b_a, b.segment<3>(3)));
  EXPECT(veq(xi.b_v, b.segment<3>(6)));
}

// ---------------------------------------------------------------------------
// traits::Equals
// ---------------------------------------------------------------------------

TEST(State, EqualsIdentities) {
  EXPECT(traits<State>::Equals(State::identity(), State::identity(), kTol));
}

TEST(State, EqualsDistinguishesStates) {
  State a = State::identity();
  State b = State::identity();
  b.v = Eigen::Vector3d(1.0, 0.0, 0.0);
  EXPECT(!traits<State>::Equals(a, b, kTol));
}

// ---------------------------------------------------------------------------
// traits::Retract / Local round-trips
// ---------------------------------------------------------------------------

TEST(State, RetractZeroDeltaIsNoop) {
  State xi;
  xi.R = Rot3::Ry(0.3);
  xi.v = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.p = Eigen::Vector3d(0.1, 0.2, 0.3);
  xi.b_w = Eigen::Vector3d(0.01, 0.02, 0.03);
  xi.b_a = Eigen::Vector3d::Zero();
  xi.b_v = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 18, 1> zero = Eigen::Matrix<double, 18, 1>::Zero();
  State result = traits<State>::Retract(xi, zero);

  EXPECT(traits<State>::Equals(xi, result, kTol));
}

TEST(State, LocalRetractRoundTripFromIdentity) {
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.1, -0.2, 0.05, 1.0, 2.0, 3.0, -1.0, 0.5, 0.2, 0.01, 0.02, 0.03,
      -0.01, 0.0, 0.01, 0.0, -0.02, 0.01;

  State xi_ref = State::identity();
  State xi_ret = traits<State>::Retract(xi_ref, delta);
  auto delta_rec = traits<State>::Local(xi_ref, xi_ret);

  EXPECT(assert_equal((Vector)delta, (Vector)delta_rec, kTol));
}

TEST(State, RetractLocalRoundTripFromNonIdentity) {
  State xi_ref;
  xi_ref.R = Rot3::Rz(0.5) * Rot3::Rx(0.3);
  xi_ref.v = Eigen::Vector3d(10.0, -5.0, 2.0);
  xi_ref.p = Eigen::Vector3d(1.0, 0.5, -0.2);
  xi_ref.b_w = Eigen::Vector3d(0.01, -0.01, 0.02);
  xi_ref.b_a = Eigen::Vector3d(-0.1, 0.0, 0.1);
  xi_ref.b_v = Eigen::Vector3d(0.05, 0.0, -0.05);

  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.05, -0.1, 0.02, 0.5, -0.5, 1.0, -0.1, 0.2, 0.0, 0.005, 0.0, -0.005,
      0.0, 0.01, 0.0, -0.01, 0.0, 0.01;

  State xi_pert = traits<State>::Retract(xi_ref, delta);
  auto delta_rec = traits<State>::Local(xi_ref, xi_pert);

  EXPECT(assert_equal((Vector)delta, (Vector)delta_rec, 1e-7));
}

// ---------------------------------------------------------------------------
// traits::Print (smoke — just check no crash)
// ---------------------------------------------------------------------------

TEST(State, PrintDoesNotCrash) {
  traits<State>::Print(State::identity(), "identity: ");
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
