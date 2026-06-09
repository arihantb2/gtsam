#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/State.h>

using namespace tgeqf;
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
// TGState::identity
// ---------------------------------------------------------------------------

// Checks that identity() returns R=I_3 and all six 3-vectors zero.
TEST(TGState, IdentityIsZero) {
  TGState xi = TGState::identity();
  EXPECT(xi.R.equals(Rot3::Identity(), kTol));
  EXPECT(veq(xi.v,   Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.p,   Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_w, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_a, Eigen::Vector3d::Zero()));
  EXPECT(veq(xi.b_v, Eigen::Vector3d::Zero()));
}

// ---------------------------------------------------------------------------
// TGState::T_matrix
// ---------------------------------------------------------------------------

// Checks that T_matrix() at the identity state equals the 5x5 identity matrix.
TEST(TGState, TMatrixAtIdentityIsI5) {
  auto T = TGState::identity().T_matrix();
  EXPECT(meq(T, Eigen::Matrix<double, 5, 5>::Identity()));
}

// Checks that T_matrix() places R in (0:3,0:3), v in col 3, p in col 4,
// 1s at (3,3) and (4,4), and zeros in the bottom-left 2x3 padding block.
TEST(TGState, TMatrixBlocks) {
  TGState xi;
  xi.R       = Rot3::Rx(M_PI / 2.0);
  xi.v       = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.p       = Eigen::Vector3d(4.0, 5.0, 6.0);

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
// TGState::bias_vector
// ---------------------------------------------------------------------------

// Checks that bias_vector() packs b_w at [0:3], b_a at [3:6], b_v at [6:9].
TEST(TGState, BiasVectorPacking) {
  TGState xi       = TGState::identity();
  xi.b_w     = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.b_a     = Eigen::Vector3d(4.0, 5.0, 6.0);
  xi.b_v     = Eigen::Vector3d(7.0, 8.0, 9.0);

  auto b = xi.bias_vector();

  EXPECT(veq(xi.b_w,     b.segment<3>(0)));
  EXPECT(veq(xi.b_a,     b.segment<3>(3)));
  EXPECT(veq(xi.b_v,     b.segment<3>(6)));
}

// ---------------------------------------------------------------------------
// traits::Equals
// ---------------------------------------------------------------------------

// Checks that Equals returns true when both states are the identity.
TEST(TGState, EqualsIdentities) {
  EXPECT(traits<TGState>::Equals(TGState::identity(), TGState::identity(), kTol));
}

// Checks that Equals returns false when two states differ only in velocity,
// confirming all fields contribute to equality.
TEST(TGState, EqualsDistinguishesStates) {
  TGState a = TGState::identity();
  TGState b = TGState::identity();
  b.v       = Eigen::Vector3d(1.0, 0.0, 0.0);
  EXPECT(!traits<TGState>::Equals(a, b, kTol));
}

// ---------------------------------------------------------------------------
// traits::Retract / Local round-trips
// ---------------------------------------------------------------------------

// Checks that Retract(xi, 0) == xi for a non-identity state;
// a zero tangent perturbation must leave the state unchanged.
TEST(TGState, RetractZeroDeltaIsNoop) {
  TGState xi;
  xi.R       = Rot3::Ry(0.3);
  xi.v       = Eigen::Vector3d(1.0, 2.0, 3.0);
  xi.p       = Eigen::Vector3d(0.1, 0.2, 0.3);
  xi.b_w     = Eigen::Vector3d(0.01, 0.02, 0.03);
  xi.b_a     = Eigen::Vector3d::Zero();
  xi.b_v     = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 18, 1> zero = Eigen::Matrix<double, 18, 1>::Zero();
  TGState result = traits<TGState>::Retract(xi, zero);

  EXPECT(traits<TGState>::Equals(xi, result, kTol));
}

// Checks Local(ref, Retract(ref, delta)) == delta when ref is the identity;
// verifies the Retract/Local inverse relationship at the origin.
TEST(TGState, LocalRetractRoundTripFromIdentity) {
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.1, -0.2, 0.05,
           1.0,  2.0,  3.0,
          -1.0,  0.5,  0.2,
           0.01, 0.02, 0.03,
          -0.01, 0.0,  0.01,
           0.0, -0.02, 0.01;

  TGState xi_ref = TGState::identity();
  TGState xi_ret = traits<TGState>::Retract(xi_ref, delta);
  auto delta_rec = traits<TGState>::Local(xi_ref, xi_ret);

  EXPECT(assert_equal((Vector)delta, (Vector)delta_rec, kTol));
}

// Checks Local(ref, Retract(ref, delta)) == delta for a non-identity reference;
// verifies the Retract/Local inverse relationship away from the origin.
TEST(TGState, RetractLocalRoundTripFromNonIdentity) {
  TGState xi_ref;
  xi_ref.R       = Rot3::Rz(0.5) * Rot3::Rx(0.3);
  xi_ref.v       = Eigen::Vector3d(10.0, -5.0, 2.0);
  xi_ref.p       = Eigen::Vector3d(1.0, 0.5, -0.2);
  xi_ref.b_w     = Eigen::Vector3d(0.01, -0.01, 0.02);
  xi_ref.b_a     = Eigen::Vector3d(-0.1, 0.0, 0.1);
  xi_ref.b_v     = Eigen::Vector3d(0.05, 0.0, -0.05);

  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.05, -0.1,  0.02,
           0.5,  -0.5,  1.0,
          -0.1,   0.2,  0.0,
           0.005, 0.0, -0.005,
           0.0,   0.01, 0.0,
          -0.01,  0.0,  0.01;

  TGState xi_pert = traits<TGState>::Retract(xi_ref, delta);
  auto delta_rec  = traits<TGState>::Local(xi_ref, xi_pert);

  EXPECT(assert_equal((Vector)delta, (Vector)delta_rec, 1e-7));
}

// ---------------------------------------------------------------------------
// traits::Print (smoke — just check no crash)
// ---------------------------------------------------------------------------

// Checks that Print executes without throwing for the identity state.
TEST(TGState, PrintDoesNotCrash) {
  traits<TGState>::Print(TGState::identity(), "identity: ");
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
