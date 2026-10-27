/**
 * @file  testTGGroup.cpp
 * @brief Group axioms for G_TG = TangentLieGroup<ExtendedPose3<2>>, the
 *        tangent group of SE_2(3) under its adjoint action, and for the
 *        filter's symmetry group G_TG ⋉ R^3 that adds the magnetometer bias.
 *
 * Layer above testTGState.cpp: the symmetry group the observer state lives
 * on. TGElement is a plain alias for gtsam's generic TangentLieGroup<G>, so
 * there is no bespoke algebra left here to test (that lives in
 * tests/testTangentLieGroup.cpp and gtsam/geometry/tests/testExtendedPose3.cpp
 * upstream); these are axioms confirming the alias is correctly wired into
 * the traits<TGElement> Lie-group machinery (associativity, inverse,
 * homomorphism, exp/log round trip).
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/TGGeometry.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;  // looser, for composed operations

bool meq(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
         double tol = kTol) {
  return assert_equal((Matrix)a, (Matrix)b, tol);
}

TGElement makeElement(const Rot3& R, const Vector3& v, const Vector3& p,
                      const Vector9& a) {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = v;
  vp.col(1) = p;
  return TGElement(ExtendedPose3<2>(R, vp), a);
}

TGElement makeX() {
  return makeElement(
      Rot3::Rz(0.4) * Rot3::Rx(0.2), Vector3(1.0, -2.0, 0.5),
      Vector3(0.3, 0.1, -0.4),
      (Vector9() << 0.1, -0.1, 0.05, -0.2, 0.3, 0.0, 0.0, 0.1, -0.1)
          .finished());
}

TGElement makeY() {
  return makeElement(
      Rot3::Ry(0.3) * Rot3::Rz(-0.1), Vector3(0.2, -0.3, 0.1),
      Vector3(-0.5, 1.0, 2.0),
      (Vector9() << 0.05, 0.0, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.05)
          .finished());
}

TGElement makeZ() {
  return makeElement(
      Rot3::Rx(0.5), Vector3(0.0, 0.0, 1.0), Vector3(0.2, -0.1, 0.3),
      (Vector9() << 0.03, -0.02, 0.01, -0.05, 0.0, 0.02, 0.0, -0.03, 0.04)
          .finished());
}

Eigen::Matrix<double, 18, 1> makeDelta() {
  Eigen::Matrix<double, 18, 1> delta;
  delta << 0.05, -0.1, 0.03, 0.2, -0.3, 0.1, -0.1, 0.2, -0.05, 0.01, 0.02,
      -0.01, 0.05, 0.0, -0.05, 0.0, 0.03, -0.02;
  return delta;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace group_axioms {

// The identity is neutral on both sides.
TEST(TGElement, IdentityIsNeutral) {
  const TGElement X = fixture::makeX();
  const TGElement I = TGElement::Identity();
  EXPECT(traits<TGElement>::Equals(I * X, X, fixture::kTol));
  EXPECT(traits<TGElement>::Equals(X * I, X, fixture::kTol));
}

// inverse() is a two-sided inverse.
TEST(TGElement, InverseIsTwoSided) {
  const TGElement X = fixture::makeX();
  const TGElement XI = X.inverse();
  EXPECT(
      traits<TGElement>::Equals(XI * X, TGElement::Identity(), fixture::kTol));
  EXPECT(
      traits<TGElement>::Equals(X * XI, TGElement::Identity(), fixture::kTol));
}

// The semidirect product is associative. This is the axiom the fiber term
// c + Ad_A[d] has to satisfy.
TEST(TGElement, ProductIsAssociative) {
  const TGElement X = fixture::makeX(), Y = fixture::makeY(),
                  Z = fixture::makeZ();
  EXPECT(traits<TGElement>::Equals((X * Y) * Z, X * (Y * Z), fixture::kTolL));
}

// The 18x18 AdjointMap is a homomorphism and reduces to the identity at id.
TEST(TGElement, AdjointMapIsAHomomorphism) {
  const TGElement X = fixture::makeX(), Y = fixture::makeY();
  using Tr = traits<TGElement>;

  EXPECT(fixture::meq(Tr::AdjointMap(X * Y),
                      Tr::AdjointMap(X) * Tr::AdjointMap(Y), fixture::kTolL));
  EXPECT(fixture::meq(Tr::AdjointMap(TGElement::Identity()),
                      Eigen::Matrix<double, 18, 18>::Identity(),
                      fixture::kTol));
}

}  // namespace group_axioms
/* ************************************************************************* */

/* ************************************************************************* */
namespace exponential {

// Exp and Log agree at the identity and invert each other away from it.
TEST(TGElement, ExpmapLogmapRoundTrip) {
  EXPECT(traits<TGElement>::Equals(
      TGElement::Expmap(Eigen::Matrix<double, 18, 1>::Zero()),
      TGElement::Identity(), fixture::kTol));
  EXPECT(assert_equal((Vector)Eigen::Matrix<double, 18, 1>::Zero(),
                      (Vector)TGElement::Logmap(TGElement::Identity()),
                      fixture::kTol));

  const Eigen::Matrix<double, 18, 1> v = fixture::makeDelta();
  EXPECT(assert_equal((Vector)v,
                      (Vector)TGElement::Logmap(TGElement::Expmap(v)),
                      fixture::kTolL));
}

// Exp intertwines the adjoint: Exp(Ad_X w) = X Exp(w) X^-1.
TEST(TGElement, ExpmapIntertwinesTheAdjoint) {
  const TGElement X = fixture::makeX();
  const Eigen::Matrix<double, 18, 1> w =
      0.1 * Eigen::Matrix<double, 18, 1>::Ones();

  EXPECT(traits<TGElement>::Equals(
      TGElement::Expmap(traits<TGElement>::AdjointMap(X) * w),
      X * TGElement::Expmap(w) * X.inverse(), fixture::kTol));
}

// The right chart round-trips, at the identity and away from it.
TEST(TGElement, RetractLocalRoundTrip) {
  const Eigen::Matrix<double, 18, 1> delta = fixture::makeDelta();

  for (const TGElement& X : {TGElement::Identity(), fixture::makeX()}) {
    const TGElement Y = traits<TGElement>::Retract(X, delta);
    EXPECT(assert_equal((Vector)delta, (Vector)traits<TGElement>::Local(X, Y),
                        fixture::kTolL));
  }
}

// The Ad-based Compose/Between/Inverse Jacobians are the true derivatives in
// the traits chart.
TEST(TGElement, ChartJacobiansMatchNumerical) {
  const TGElement X = fixture::makeX(), Y = fixture::makeY();
  using Tr = traits<TGElement>;
  const auto compose = [](const TGElement& a, const TGElement& b) {
    return a * b;
  };
  const auto between = [](const TGElement& a, const TGElement& b) {
    return a.inverse() * b;
  };
  Eigen::Matrix<double, 18, 18> H1, H2;

  Tr::Compose(X, Y, H1, H2);
  EXPECT(fixture::meq(numericalDerivative21(compose, X, Y), H1, 1e-6));
  EXPECT(fixture::meq(numericalDerivative22(compose, X, Y), H2, 1e-6));

  Tr::Inverse(X, H1);
  EXPECT(fixture::meq(
      numericalDerivative11([](const TGElement& a) { return a.inverse(); }, X),
      H1, 1e-6));

  Tr::Between(X, Y, H1, H2);
  EXPECT(fixture::meq(numericalDerivative21(between, X, Y), H1, 1e-6));
  EXPECT(fixture::meq(numericalDerivative22(between, X, Y), H2, 1e-6));

  // Expmap/Logmap also carry real Jacobians (a generic matrix-exponential
  // kernel, since ExtendedPose3<2> has no specialized closed form), unlike
  // the hand-rolled TGElement this type replaces.
  const Eigen::Matrix<double, 18, 1> v = fixture::makeDelta();
  Tr::Expmap(v, H1);
  EXPECT(fixture::meq(
      numericalDerivative11<TGElement, Eigen::Matrix<double, 18, 1>>(
          [](const Eigen::Matrix<double, 18, 1>& u) {
            return TGElement::Expmap(u);
          },
          v),
      H1, 1e-6));

  Tr::Logmap(X, H1);
  EXPECT(fixture::meq(
      numericalDerivative11<Eigen::Matrix<double, 18, 1>, TGElement>(
          [](const TGElement& g) { return TGElement::Logmap(g); }, X),
      H1, 1e-6));
}

}  // namespace exponential
/* ************************************************************************* */

/* ************************************************************************* */
namespace mag_group {

using Tangent21 = Eigen::Matrix<double, 21, 1>;

TGMagElement makeX() {
  return TGMagElement(fixture::makeX(), Vector3(0.2, -0.1, 0.3));
}

TGMagElement makeY() {
  return TGMagElement(fixture::makeY(), Vector3(-0.4, 0.05, 0.1));
}

TGMagElement makeZ() {
  return TGMagElement(fixture::makeZ(), Vector3(0.1, 0.2, -0.15));
}

Tangent21 makeDelta() {
  Tangent21 delta;
  delta << fixture::makeDelta(), 0.04, -0.03, 0.02;
  return delta;
}

// The product is (A1 A2, a1 + Ad_{A1} a2, d1 + R_{A1} d2), the law the
// magnetometer-bias action R_A^T (bm - d) needs to be a right action.
TEST(TGMagElement, ProductMatchesTheSemidirectLaw) {
  const TGMagElement X = makeX(), Y = makeY();
  const TGMagElement XY = X * Y;

  EXPECT(traits<TGElement>::Equals(X.first * Y.first, XY.first, fixture::kTol));
  EXPECT(assert_equal(
      (Vector)(X.second + X.first.first.rotation().rotate(Y.second)),
      (Vector)XY.second, fixture::kTol));
}

// The identity is neutral and inverse() is a two-sided inverse.
TEST(TGMagElement, IdentityAndInverse) {
  const TGMagElement X = makeX();
  const TGMagElement I = TGMagElement::Identity();
  EXPECT(traits<TGMagElement>::Equals(I * X, X, fixture::kTol));
  EXPECT(traits<TGMagElement>::Equals(X * I, X, fixture::kTol));
  EXPECT(traits<TGMagElement>::Equals(X.inverse() * X, I, fixture::kTol));
  EXPECT(traits<TGMagElement>::Equals(X * X.inverse(), I, fixture::kTol));
}

// The product is associative.
TEST(TGMagElement, ProductIsAssociative) {
  const TGMagElement X = makeX(), Y = makeY(), Z = makeZ();
  EXPECT(
      traits<TGMagElement>::Equals((X * Y) * Z, X * (Y * Z), fixture::kTolL));
}

// The 21x21 AdjointMap is a homomorphism and reduces to the identity at id.
TEST(TGMagElement, AdjointMapIsAHomomorphism) {
  const TGMagElement X = makeX(), Y = makeY();
  using Tr = traits<TGMagElement>;

  EXPECT(fixture::meq(Tr::AdjointMap(X * Y),
                      Tr::AdjointMap(X) * Tr::AdjointMap(Y), fixture::kTolL));
  EXPECT(fixture::meq(Tr::AdjointMap(TGMagElement::Identity()),
                      Eigen::Matrix<double, 21, 21>::Identity(),
                      fixture::kTol));
}

// Exp and Log invert each other, and Exp intertwines the adjoint.
TEST(TGMagElement, ExpmapLogmapRoundTrip) {
  const Tangent21 v = makeDelta();
  EXPECT(assert_equal((Vector)v,
                      (Vector)TGMagElement::Logmap(TGMagElement::Expmap(v)),
                      fixture::kTolL));

  const TGMagElement X = makeX();
  EXPECT(traits<TGMagElement>::Equals(
      TGMagElement::Expmap(traits<TGMagElement>::AdjointMap(X) * v),
      X * TGMagElement::Expmap(v) * X.inverse(), fixture::kTolL));
}

// Dropping d is a homomorphism onto the tangent group, so the navigation and
// IMU-bias parts of the filter evolve exactly as without the magnetometer.
TEST(TGMagElement, ProjectionToTheTangentGroupIsAHomomorphism) {
  const TGMagElement X = makeX(), Y = makeY();
  EXPECT(traits<TGElement>::Equals((X * Y).first, X.first * Y.first,
                                   fixture::kTol));
  EXPECT(traits<TGElement>::Equals(X.inverse().first, X.first.inverse(),
                                   fixture::kTol));

  const Tangent21 v = makeDelta();
  EXPECT(traits<TGElement>::Equals(TGMagElement::Expmap(v).first,
                                   TGElement::Expmap(v.head<18>()),
                                   fixture::kTolL));
}

// The action R_A d carries analytic Jacobians for the right perturbation of
// X and for d, which AdjointMap() reads at the identity.
TEST(MagBiasAction, JacobiansMatchNumerical) {
  const TGElement X = fixture::makeX();
  const Vector3 d(0.3, -0.2, 0.5);
  const MagBiasAction action;

  Eigen::Matrix<double, 3, 18> HX;
  Matrix3 Hd;
  const Vector3 result = action(X, d, HX, Hd);
  EXPECT(assert_equal((Vector)X.first.rotation().rotate(d), (Vector)result,
                      fixture::kTol));

  const auto act = [&action](const TGElement& g, const Vector3& h) {
    return action(g, h);
  };
  EXPECT(fixture::meq(
      numericalDerivative21<Vector3, TGElement, Vector3>(act, X, d), HX, 1e-6));
  EXPECT(fixture::meq(
      numericalDerivative22<Vector3, TGElement, Vector3>(act, X, d), Hd, 1e-6));
}

}  // namespace mag_group
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
