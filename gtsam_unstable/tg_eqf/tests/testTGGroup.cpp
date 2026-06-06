#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/Group.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol  = 1e-9;
static constexpr double kTolL = 1e-7;  // looser for composed operations

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

// Factory: group element with known non-trivial values.
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

static TGGroupElement makeY() {
  TGGroupElement Y;
  Y.R_X = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.p_X = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.v_X = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.a   = {Eigen::Vector3d(0.05, 0.0, -0.05),
            Eigen::Vector3d(0.1, -0.1, 0.2),
            Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

// ---------------------------------------------------------------------------
// se2_3 basic operations
// ---------------------------------------------------------------------------

// Checks that vector() followed by from_vector() recovers all three components.
TEST(se2_3, VectorRoundTrip) {
  se2_3 xi = {Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6),
              Eigen::Vector3d(7, 8, 9)};
  EXPECT(veq(xi.omega,   se2_3::from_vector(xi.vector()).omega));
  EXPECT(veq(xi.v_tilde, se2_3::from_vector(xi.vector()).v_tilde));
  EXPECT(veq(xi.accel, se2_3::from_vector(xi.vector()).accel));
}

// Checks that wedge() followed by vee() recovers the original se_2(3) element.
TEST(se2_3, WedgeVeeRoundTrip) {
  se2_3 xi = {Eigen::Vector3d(0.1, -0.2, 0.3),
              Eigen::Vector3d(1.0, 2.0, 3.0),
              Eigen::Vector3d(-1.0, 0.5, 0.2)};
  se2_3 recovered = se2_3::vee(xi.wedge());
  EXPECT(veq(xi.omega,   recovered.omega));
  EXPECT(veq(xi.v_tilde, recovered.v_tilde));
  EXPECT(veq(xi.accel, recovered.accel));
}

// Checks that wedge() places omega in the 3x3 skew block, v_tilde and accel
// in columns 3 and 4, and zeros in the bottom two rows.
TEST(se2_3, WedgeStructure) {
  se2_3 xi = {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
              Eigen::Vector3d(0, 0, 1)};
  auto W = xi.wedge();
  // Rotation block is skew of omega
  EXPECT(meq(W.block<3,3>(0,0), Rot3::Hat(xi.omega)));
  // Translation columns
  EXPECT(veq(xi.v_tilde, W.block<3,1>(0,3)));
  EXPECT(veq(xi.accel, W.block<3,1>(0,4)));
  // Bottom rows zero
  EXPECT(meq(W.block<2,5>(3,0), Eigen::Matrix<double,2,5>::Zero()));
}

// ---------------------------------------------------------------------------
// TGGroupElement::Identity
// ---------------------------------------------------------------------------

// Checks that Identity() has R_X=I, all translation vectors zero, and
// all fiber components (a.omega, a.v_tilde, a.accel) zero.
TEST(TGGroupElement, Identity) {
  auto I = TGGroupElement::Identity();
  EXPECT(I.R_X.equals(Rot3::Identity(), kTol));
  EXPECT(veq(I.p_X, Eigen::Vector3d::Zero()));
  EXPECT(veq(I.v_X, Eigen::Vector3d::Zero()));
  EXPECT(veq(I.a.omega,   Eigen::Vector3d::Zero()));
  EXPECT(veq(I.a.v_tilde, Eigen::Vector3d::Zero()));
  EXPECT(veq(I.a.accel, Eigen::Vector3d::Zero()));
}

// ---------------------------------------------------------------------------
// TGGroupElement::AT_matrix
// ---------------------------------------------------------------------------

// Checks that AT_matrix() at the group identity returns the 5x5 identity matrix.
TEST(TGGroupElement, ATMatrixAtIdentityIsI5) {
  auto T = TGGroupElement::Identity().A_matrix();
  EXPECT(meq(T, Eigen::Matrix<double,5,5>::Identity()));
}

// Checks that AT_matrix() places R_X in (0:3,0:3), p_X in col 3, v_X in col 4,
// 1s at (3,3) and (4,4), and zeros in the bottom-left 2x3 block.
TEST(TGGroupElement, ATMatrixBlocks) {
  auto X = makeX();
  auto T = X.A_matrix();
  EXPECT(meq(X.R_X.matrix(), T.block<3,3>(0,0)));
  EXPECT(veq(X.p_X, T.block<3,1>(0,3)));
  EXPECT(veq(X.v_X, T.block<3,1>(0,4)));
  DOUBLES_EQUAL(1.0, T(3,3), kTol);
  DOUBLES_EQUAL(1.0, T(4,4), kTol);
  DOUBLES_EQUAL(0.0, T(3,4), kTol);
  DOUBLES_EQUAL(0.0, T(4,3), kTol);
  // bottom-left 2x3 block (rotation coupling) must be zero
  EXPECT(meq(T.block<2,3>(3,0), Eigen::Matrix<double,2,3>::Zero()));
}

// ---------------------------------------------------------------------------
// Group product and inverse
// ---------------------------------------------------------------------------

// Checks that I * X == X, confirming Identity() is a left neutral element.
TEST(TGGroupElement, IdentityIsLeftIdentity) {
  auto X = makeX();
  auto I = TGGroupElement::Identity();
  EXPECT(traits<TGGroupElement>::Equals(I * X, X, kTol));
}

// Checks that X * I == X, confirming Identity() is a right neutral element.
TEST(TGGroupElement, IdentityIsRightIdentity) {
  auto X = makeX();
  auto I = TGGroupElement::Identity();
  EXPECT(traits<TGGroupElement>::Equals(X * I, X, kTol));
}

// Checks that X^{-1} * X == I (left inverse).
TEST(TGGroupElement, InverseTimesXIsIdentity) {
  auto X  = makeX();
  auto XI = X.inverse();
  EXPECT(traits<TGGroupElement>::Equals(XI * X, TGGroupElement::Identity(), kTol));
}

// Checks that X * X^{-1} == I (right inverse).
TEST(TGGroupElement, XTimesInverseIsIdentity) {
  auto X  = makeX();
  auto XI = X.inverse();
  EXPECT(traits<TGGroupElement>::Equals(X * XI, TGGroupElement::Identity(), kTol));
}

// Checks that (X * Y) * Z == X * (Y * Z) for distinct group elements;
// verifies the semidirect-product composition is associative.
TEST(TGGroupElement, GroupProductAssociativity) {
  auto X = makeX(), Y = makeY(), Z = makeX();  // reuse X as Z
  Z.R_X = Rot3::Rx(0.5);
  Z.p_X = Eigen::Vector3d(0, 0, 1);
  EXPECT(traits<TGGroupElement>::Equals((X * Y) * Z, X * (Y * Z), kTolL));
}

// ---------------------------------------------------------------------------
// Adjoint maps
// ---------------------------------------------------------------------------

// Checks that Ad_A_inv(Ad_A(xi)) == xi, confirming the two adjoints are
// exact inverses of each other on se_2(3).
TEST(TGGroupElement, AdATAndAdATInvAreInverses) {
  auto X  = makeX();
  se2_3 xi = {Eigen::Vector3d(0.1,-0.2,0.3),
               Eigen::Vector3d(1,2,3), Eigen::Vector3d(-1,0,0.5)};
  se2_3 round_trip = X.Ad_A_inv(X.Ad_A(xi));
  EXPECT(veq(xi.omega,   round_trip.omega,   kTolL));
  EXPECT(veq(xi.v_tilde, round_trip.v_tilde, kTolL));
  EXPECT(veq(xi.accel, round_trip.accel, kTolL));
}

// Checks that Ad_{XY}[xi] == Ad_X[Ad_Y[xi]], confirming Ad_A is a group
// homomorphism from G_TG into GL(se_2(3)).
TEST(TGGroupElement, AdATSatisfiesGroupProductHomomorphism) {
  // Ad_{XY}[xi] == Ad_X[Ad_Y[xi]]
  auto X  = makeX(), Y = makeY();
  se2_3 xi = {Eigen::Vector3d(0.2,0.1,-0.1),
               Eigen::Vector3d(0.5,-1,0), Eigen::Vector3d(0,0.3,-0.2)};
  auto lhs = (X * Y).Ad_A(xi);
  auto rhs = X.Ad_A(Y.Ad_A(xi));
  EXPECT(veq(lhs.omega,   rhs.omega,   kTolL));
  EXPECT(veq(lhs.v_tilde, rhs.v_tilde, kTolL));
  EXPECT(veq(lhs.accel, rhs.accel, kTolL));
}

// ---------------------------------------------------------------------------
// Expmap / Logmap
// ---------------------------------------------------------------------------

// Checks that Expmap(0) returns the group identity.
TEST(TGGroupElement, ExpmapAtZeroIsIdentity) {
  auto X = TGGroupElement::Expmap(Eigen::Matrix<double,18,1>::Zero());
  EXPECT(traits<TGGroupElement>::Equals(X, TGGroupElement::Identity(), kTol));
}

// Checks that Identity().Logmap() returns the zero 18-vector.
TEST(TGGroupElement, LogmapOfIdentityIsZero) {
  auto v = TGGroupElement::Identity().Logmap();
  EXPECT(assert_equal((Vector)Eigen::Matrix<double,18,1>::Zero(), (Vector)v, kTol));
}

// Checks that Logmap(Expmap(v)) == v for a small tangent vector;
// the SE_2(3) part is exact, and the fiber uses the first-order approximation Ξ≈I.
TEST(TGGroupElement, ExpmapLogmapRoundTrip) {
  Eigen::Matrix<double,18,1> v;
  v << 0.1, -0.2, 0.05,   // tau_omega (small for first-order accuracy)
       0.3, -0.5,  0.2,   // tau_eta
      -0.1,  0.4, -0.2,   // tau_alpha
       0.05, 0.1, -0.05,  // sigma_omega
      -0.2,  0.3,  0.0,   // sigma_v
       0.0, -0.1,  0.2;   // sigma_a
  auto v_rec = TGGroupElement::Expmap(v).Logmap();
  EXPECT(assert_equal((Vector)v, (Vector)v_rec, kTolL));
}

// ---------------------------------------------------------------------------
// traits::Retract / Local round-trip
// ---------------------------------------------------------------------------

// Checks Local(I, Retract(I, delta)) == delta from the group identity,
// verifying the right-Retract/Local inverse relationship at the origin.
TEST(TGGroupElement, LocalRetractRoundTripFromIdentity) {
  Eigen::Matrix<double,18,1> delta;
  delta << 0.05, -0.1, 0.03, 0.2, -0.3, 0.1,
           -0.1, 0.2, -0.05, 0.01, 0.02, -0.01,
           0.05, 0.0, -0.05, 0.0, 0.03, -0.02;

  auto I   = TGGroupElement::Identity();
  auto X   = traits<TGGroupElement>::Retract(I, delta);
  auto rec = traits<TGGroupElement>::Local(I, X);
  EXPECT(assert_equal((Vector)delta, (Vector)rec, kTolL));
}

// Checks Local(X, Retract(X, delta)) == delta from a non-identity base point,
// verifying the right-Retract/Local inverse relationship away from the origin.
TEST(TGGroupElement, LocalRetractRoundTripFromNonIdentity) {
  auto X = makeX();
  Eigen::Matrix<double,18,1> delta;
  delta << 0.02, -0.05, 0.01, 0.1, -0.1, 0.05,
           -0.05, 0.1, 0.0, 0.005, 0.0, -0.005,
           0.01, -0.01, 0.0, 0.0, 0.02, -0.01;

  auto Y   = traits<TGGroupElement>::Retract(X, delta);
  auto rec = traits<TGGroupElement>::Local(X, Y);
  EXPECT(assert_equal((Vector)delta, (Vector)rec, kTolL));
}

// ---------------------------------------------------------------------------
// traits::AdjointMap structure
// ---------------------------------------------------------------------------

// Checks that the 18x18 AdjointMap has the correct block structure:
// the top-right 9x9 block is zero, and the two diagonal 9x9 blocks are equal
// (both equal to Ad_C of the SE_2(3) part).
TEST(TGGroupElement, AdjointMapBlockStructure) {
  auto X   = makeX();
  auto Adj = traits<TGGroupElement>::AdjointMap(X);

  // Top-right block must be zero
  EXPECT(meq(Adj.block<9,9>(0,9), Eigen::Matrix<double,9,9>::Zero()));
  // Diagonal blocks must be equal (both Ad_C)
  EXPECT(meq(Adj.block<9,9>(0,0), Adj.block<9,9>(9,9)));
}

// Checks that AdjointMap(Identity) == I_18, since Ad_I = Id on the algebra.
TEST(TGGroupElement, AdjointMapAtIdentityIsIdentity18) {
  auto I   = TGGroupElement::Identity();
  auto Adj = traits<TGGroupElement>::AdjointMap(I);
  EXPECT(meq(Adj, Eigen::Matrix<double,18,18>::Identity()));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
