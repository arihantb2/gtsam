/**
 * @file  testTGGroup.cpp
 * @brief Unit tests for the tangent-group element G_TG = SE_2(3) ltimes
 * se_2(3): se2_3 fiber ops, group product/inverse, Adjoint, Expmap/Logmap.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Group.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-9;
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
static TGElement makeX() {
  TGElement X;
  X.R = Rot3::Rz(0.4) * Rot3::Rx(0.2);
  X.v = Eigen::Vector3d(1.0, -2.0, 0.5);
  X.p = Eigen::Vector3d(0.3, 0.1, -0.4);
  X.a = {Eigen::Vector3d(0.1, -0.1, 0.05), Eigen::Vector3d(-0.2, 0.3, 0.0),
         Eigen::Vector3d(0.0, 0.1, -0.1)};
  return X;
}

static TGElement makeY() {
  TGElement Y;
  Y.R = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.p = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.v = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.a = {Eigen::Vector3d(0.05, 0.0, -0.05), Eigen::Vector3d(0.1, -0.1, 0.2),
         Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

// ---------------------------------------------------------------------------
// se2_3 basic operations
// ---------------------------------------------------------------------------

TEST(se2_3, VectorRoundTrip) {
  se2_3 a = {Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6),
             Eigen::Vector3d(7, 8, 9)};
  EXPECT(veq(a.w, se2_3::from_vector(a.vector()).w));
  EXPECT(veq(a.a, se2_3::from_vector(a.vector()).a));
  EXPECT(veq(a.v, se2_3::from_vector(a.vector()).v));
}

TEST(se2_3, WedgeVeeRoundTrip) {
  se2_3 a = {Eigen::Vector3d(0.1, -0.2, 0.3), Eigen::Vector3d(1.0, 2.0, 3.0),
             Eigen::Vector3d(-1.0, 0.5, 0.2)};
  se2_3 recovered = se2_3::vee(a.wedge());
  EXPECT(veq(a.w, recovered.w));
  EXPECT(veq(a.a, recovered.a));
  EXPECT(veq(a.v, recovered.v));
}

TEST(se2_3, WedgeStructure) {
  se2_3 a = {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
             Eigen::Vector3d(0, 0, 1)};
  auto W = a.wedge();
  // Rotation block is skew of omega
  EXPECT(meq(W.block<3, 3>(0, 0), Rot3::Hat(a.w)));
  // Translation columns
  EXPECT(veq(a.a, W.block<3, 1>(0, 3)));
  EXPECT(veq(a.v, W.block<3, 1>(0, 4)));
  // Bottom rows zero
  EXPECT(meq(W.block<2, 5>(3, 0), Eigen::Matrix<double, 2, 5>::Zero()));
}

// ---------------------------------------------------------------------------
// TGElement::to_A_matrix
// ---------------------------------------------------------------------------

// to_A_matrix() places R in (0:3,0:3), v in col 3, p in col 4, 1s at (3,3)
// and (4,4), and zeros in the bottom-left 2x3 block.
TEST(TGElement, AMatrixBlocks) {
  auto X = makeX();
  auto A = X.to_A_matrix();
  EXPECT(meq(X.R.matrix(), A.block<3, 3>(0, 0)));
  EXPECT(veq(X.v, A.block<3, 1>(0, 3)));
  EXPECT(veq(X.p, A.block<3, 1>(0, 4)));
  DOUBLES_EQUAL(1.0, A(3, 3), kTol);
  DOUBLES_EQUAL(1.0, A(4, 4), kTol);
  DOUBLES_EQUAL(0.0, A(3, 4), kTol);
  DOUBLES_EQUAL(0.0, A(4, 3), kTol);
  // bottom-left 2x3 block must be zero
  EXPECT(meq(A.block<2, 3>(3, 0), Eigen::Matrix<double, 2, 3>::Zero()));
}

// ---------------------------------------------------------------------------
// Group product and inverse
// ---------------------------------------------------------------------------

TEST(TGElement, IdentityIsNeutral) {
  auto X = makeX();
  auto I = TGElement::Identity();
  EXPECT(traits<TGElement>::Equals(I * X, X, kTol));
  EXPECT(traits<TGElement>::Equals(X * I, X, kTol));
}

TEST(TGElement, Inverse) {
  auto X = makeX();
  auto XI = X.inverse();
  EXPECT(traits<TGElement>::Equals(XI * X, TGElement::Identity(), kTol));
  EXPECT(traits<TGElement>::Equals(X * XI, TGElement::Identity(), kTol));
}

TEST(TGElement, GroupProductAssociativity) {
  auto X = makeX(), Y = makeY();
  TGElement Z;
  Z.R = Rot3::Rx(0.5);
  Z.v = Eigen::Vector3d(0, 0, 1);
  Z.p = Eigen::Vector3d(0.2, -0.1, 0.3);
  Z.a = {Eigen::Vector3d(0.03, -0.02, 0.01), Eigen::Vector3d(-0.05, 0.0, 0.02),
         Eigen::Vector3d(0.0, -0.03, 0.04)};
  EXPECT(traits<TGElement>::Equals((X * Y) * Z, X * (Y * Z), kTolL));
}

// ---------------------------------------------------------------------------
// Adjoint maps
// ---------------------------------------------------------------------------

// Ad_A_inv(Ad_A(xi)) == xi: the two adjoints are exact inverses on se_2(3).
TEST(TGElement, AdjointRoundTrip) {
  auto X = makeX();
  se2_3 a = {Eigen::Vector3d(0.1, -0.2, 0.3), Eigen::Vector3d(1, 2, 3),
             Eigen::Vector3d(-1, 0, 0.5)};
  se2_3 round_trip = X.Ad_A_inv(X.Ad_A(a));
  EXPECT(veq(a.w, round_trip.w, kTolL));
  EXPECT(veq(a.a, round_trip.a, kTolL));
  EXPECT(veq(a.v, round_trip.v, kTolL));
}

// Ad_{XY}[xi] == Ad_X[Ad_Y[xi]]: Ad_A is a group homomorphism from G_TG into
// GL(se_2(3)).
TEST(TGElement, AdjointHomomorphism) {
  auto X = makeX(), Y = makeY();
  se2_3 a = {Eigen::Vector3d(0.2, 0.1, -0.1), Eigen::Vector3d(0.5, -1, 0),
             Eigen::Vector3d(0, 0.3, -0.2)};
  auto lhs = (X * Y).Ad_A(a);
  auto rhs = X.Ad_A(Y.Ad_A(a));
  EXPECT(veq(lhs.w, rhs.w, kTolL));
  EXPECT(veq(lhs.a, rhs.a, kTolL));
  EXPECT(veq(lhs.v, rhs.v, kTolL));
}

// ---------------------------------------------------------------------------
// Expmap / Logmap
// ---------------------------------------------------------------------------

TEST(TGElement, ExpmapAtZeroIsIdentity) {
  auto X = TGElement::Expmap(Eigen::Matrix<double, 18, 1>::Zero());
  EXPECT(traits<TGElement>::Equals(X, TGElement::Identity(), kTol));
}

TEST(TGElement, LogmapOfIdentityIsZero) {
  auto v = TGElement::Identity().Logmap();
  EXPECT(assert_equal((Vector)Eigen::Matrix<double, 18, 1>::Zero(), (Vector)v,
                      kTol));
}

// Both the SE_2(3) part and the fiber (via Xi(tau), the tangent-group dexp)
// round-trip exactly.
TEST(TGElement, ExpmapLogmapRoundTrip) {
  Eigen::Matrix<double, 18, 1> v;
  v << 0.1, -0.2, 0.05,  // dR
      0.3, -0.5, 0.2,    // dv
      -0.1, 0.4, -0.2,   // dp
      0.05, 0.1, -0.05,  // dw
      -0.2, 0.3, 0.0,    // da
      0.0, -0.1, 0.2;    // dv
  auto v_rec = TGElement::Expmap(v).Logmap();
  EXPECT(assert_equal((Vector)v, (Vector)v_rec, kTolL));
}

// Exp intertwines the adjoint: Exp(Ad_X w) == X * Exp(w) * X^{-1}. Also only
// holds for the exact exponential.
TEST(TGElement, ExpmapAdjointHomomorphism) {
  const auto X = makeX();
  Eigen::Matrix<double, 18, 1> w = 0.1 * Eigen::Matrix<double, 18, 1>::Ones();
  const auto AdX = traits<TGElement>::AdjointMap(X);
  const auto lhs = TGElement::Expmap(AdX * w);
  const auto rhs = X * TGElement::Expmap(w) * X.inverse();
  EXPECT(traits<TGElement>::Equals(lhs, rhs, 1e-9));
}

// ---------------------------------------------------------------------------
// traits::Retract / Local round-trip
// ---------------------------------------------------------------------------

TEST(TGElement, LocalRetractRoundTripFromIdentity) {
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.05, -0.1, 0.03, 0.2, -0.3, 0.1, -0.1, 0.2, -0.05, 0.01, 0.02,
      -0.01, 0.05, 0.0, -0.05, 0.0, 0.03, -0.02;

  auto I = TGElement::Identity();
  auto X = traits<TGElement>::Retract(I, delta);
  auto rec = traits<TGElement>::Local(I, X);
  EXPECT(assert_equal((Vector)delta, (Vector)rec, kTolL));
}

TEST(TGElement, LocalRetractRoundTripFromNonIdentity) {
  auto X = makeX();
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.02, -0.05, 0.01, 0.1, -0.1, 0.05, -0.05, 0.1, 0.0, 0.005, 0.0,
      -0.005, 0.01, -0.01, 0.0, 0.0, 0.02, -0.01;

  auto Y = traits<TGElement>::Retract(X, delta);
  auto rec = traits<TGElement>::Local(X, Y);
  EXPECT(assert_equal((Vector)delta, (Vector)rec, kTolL));
}

// ---------------------------------------------------------------------------
// traits::AdjointMap structure
// ---------------------------------------------------------------------------

// Top-right 9x9 block is zero; the two diagonal 9x9 blocks are equal (both
// Ad_A of the SE_2(3) part).
TEST(TGElement, AdjointMapBlockStructure) {
  auto X = makeX();
  auto Adj = traits<TGElement>::AdjointMap(X);

  // Top-right block must be zero
  EXPECT(meq(Adj.block<9, 9>(0, 9), Eigen::Matrix<double, 9, 9>::Zero()));
  // Diagonal blocks must be equal (both Ad_A)
  EXPECT(meq(Adj.block<9, 9>(0, 0), Adj.block<9, 9>(9, 9)));
}

// Ad_I = Id on the algebra.
TEST(TGElement, AdjointMapAtIdentityIsIdentity18) {
  auto I = TGElement::Identity();
  auto Adj = traits<TGElement>::AdjointMap(I);
  EXPECT(meq(Adj, Eigen::Matrix<double, 18, 18>::Identity()));
}

// ---------------------------------------------------------------------------
// traits::Compose / Between / Inverse Jacobians
// ---------------------------------------------------------------------------

// Checks that the analytic Compose/Between/Inverse Jacobians (Ad-based, right
// chart) match finite differences through the traits Retract/Local chart. The
// Ad formulas are the exact first derivatives in any chart that agrees with
// the exponential to first order at the identity, so this test is valid (and
// empirically passes) with either the exact or a first-order fiber Expmap.
TEST(TGElement, ComposeBetweenInverseJacobians) {
  const auto X = makeX();
  const auto Y = makeY();
  using Tr = traits<TGElement>;
  const auto compose = [](const TGElement& a, const TGElement& b) {
    return a * b;
  };
  const auto between = [](const TGElement& a, const TGElement& b) {
    return a.inverse() * b;
  };
  Eigen::Matrix<double, 18, 18> H1, H2;

  Tr::Compose(X, Y, H1, H2);
  EXPECT(meq(numericalDerivative21(compose, X, Y), H1, 1e-6));
  EXPECT(meq(numericalDerivative22(compose, X, Y), H2, 1e-6));

  Tr::Inverse(X, H1);
  EXPECT(meq(
      numericalDerivative11([](const TGElement& a) { return a.inverse(); }, X),
      H1, 1e-6));

  Tr::Between(X, Y, H1, H2);
  EXPECT(meq(numericalDerivative21(between, X, Y), H1, 1e-6));
  EXPECT(meq(numericalDerivative22(between, X, Y), H2, 1e-6));
}

// Must throw rather than silently leave the Jacobian uninitialized.
TEST(TGElement, ExpmapLogmapThrowOnJacobianRequest) {
  using Tr = traits<TGElement>;
  Eigen::Matrix<double, 18, 18> H;
  Eigen::Matrix<double, 18, 1> v = Eigen::Matrix<double, 18, 1>::Zero();
  CHECK_EXCEPTION(Tr::Expmap(v, H), std::runtime_error);
  CHECK_EXCEPTION(Tr::Logmap(TGElement::Identity(), H), std::runtime_error);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
