/**
 * @file  testTGGroup.cpp
 * @brief Group and algebra axioms for G_TG = SE_2(3) semidirect se_2(3).
 *
 * Layer above testTGState.cpp: the symmetry group the observer state lives on.
 * Tests are axioms (associativity, inverse, homomorphism, exp/log round trip)
 * or cross-checks between two independent implementations of the same map --
 * never a restatement of the closed form under test.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/Group.h>

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

bool aeq(const se2_3& a, const se2_3& b, double tol = kTol) {
  return assert_equal((Vector)a.vector(), (Vector)b.vector(), tol);
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

TGElement makeY() {
  TGElement Y;
  Y.R = Rot3::Ry(0.3) * Rot3::Rz(-0.1);
  Y.v = Eigen::Vector3d(0.2, -0.3, 0.1);
  Y.p = Eigen::Vector3d(-0.5, 1.0, 2.0);
  Y.a = {Eigen::Vector3d(0.05, 0.0, -0.05), Eigen::Vector3d(0.1, -0.1, 0.2),
         Eigen::Vector3d(-0.1, 0.0, 0.05)};
  return Y;
}

TGElement makeZ() {
  TGElement Z;
  Z.R = Rot3::Rx(0.5);
  Z.v = Eigen::Vector3d(0.0, 0.0, 1.0);
  Z.p = Eigen::Vector3d(0.2, -0.1, 0.3);
  Z.a = {Eigen::Vector3d(0.03, -0.02, 0.01), Eigen::Vector3d(-0.05, 0.0, 0.02),
         Eigen::Vector3d(0.0, -0.03, 0.04)};
  return Z;
}

se2_3 makeAlgebra() {
  return {Eigen::Vector3d(0.1, -0.2, 0.3), Eigen::Vector3d(1.0, 2.0, 3.0),
          Eigen::Vector3d(-1.0, 0.5, 0.2)};
}

se2_3 makeAlgebraOther() {
  return {Eigen::Vector3d(0.2, 0.1, -0.1), Eigen::Vector3d(0.5, -1.0, 0.0),
          Eigen::Vector3d(0.0, 0.3, -0.2)};
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
namespace algebra {

// vee undoes wedge, and from_vector undoes vector: the two coordinate maps on
// se_2(3) are mutual inverses.
TEST(se2_3, CoordinateMapsRoundTrip) {
  const se2_3 a = fixture::makeAlgebra();
  EXPECT(fixture::aeq(a, se2_3::vee(a.wedge())));
  EXPECT(fixture::aeq(a, se2_3::from_vector(a.vector())));
}

// ad_a is the Lie bracket in coordinates: ad_a b = vee(a^ b^ - b^ a^). Pins the
// hand-written 9x9 ad_se23 against the matrix commutator.
TEST(se2_3, AdIsTheMatrixCommutator) {
  const se2_3 a = fixture::makeAlgebra();
  const se2_3 b = fixture::makeAlgebraOther();

  const Eigen::Matrix<double, 5, 5> A = a.wedge(), B = b.wedge();
  const se2_3 bracket = se2_3::vee(A * B - B * A);

  EXPECT(assert_equal((Vector)bracket.vector(),
                      (Vector)(detail::ad_se23(a) * b.vector()),
                      fixture::kTol));
}

// Ad_A is conjugation on the algebra: (Ad_A a)^ = A a^ A^-1. Pins the closed
// form against the 5x5 matrix representation.
TEST(TGElement, AdjointIsConjugation) {
  const TGElement X = fixture::makeX();
  const se2_3 a = fixture::makeAlgebra();

  const Eigen::Matrix<double, 5, 5> A = X.to_A_matrix();
  EXPECT(fixture::meq(X.Ad_A(a).wedge(), A * a.wedge() * A.inverse(),
                      fixture::kTolL));
}

// The 9x9 Ad_SE23 / Ad_SE23_inv helpers agree with the Ad_A / Ad_A_inv closed
// forms and are mutual inverses. Both spellings are used in production.
TEST(TGElement, AdjointMatricesAgreeWithClosedForm) {
  const TGElement X = fixture::makeX();
  const se2_3 a = fixture::makeAlgebra();

  EXPECT(assert_equal((Vector)X.Ad_A(a).vector(),
                      (Vector)(detail::Ad_SE23(X) * a.vector()),
                      fixture::kTol));
  EXPECT(assert_equal((Vector)X.Ad_A_inv(a).vector(),
                      (Vector)(detail::Ad_SE23_inv(X) * a.vector()),
                      fixture::kTol));
  EXPECT(fixture::meq(detail::Ad_SE23_inv(X) * detail::Ad_SE23(X),
                      Eigen::Matrix<double, 9, 9>::Identity(), fixture::kTolL));
}

// Ad_{XY} = Ad_X Ad_Y: the adjoint is a group homomorphism.
TEST(TGElement, AdjointIsAHomomorphism) {
  const TGElement X = fixture::makeX(), Y = fixture::makeY();
  const se2_3 a = fixture::makeAlgebraOther();
  EXPECT(fixture::aeq((X * Y).Ad_A(a), X.Ad_A(Y.Ad_A(a)), fixture::kTolL));
}

}  // namespace algebra
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

// Exp and Log agree at the identity and invert each other away from it. The
// fiber leg runs through the Xi(tau) series in Exp and a solve against the same
// series in Log, so the round trip checks that the two match.
TEST(TGElement, ExpmapLogmapRoundTrip) {
  EXPECT(traits<TGElement>::Equals(
      TGElement::Expmap(Eigen::Matrix<double, 18, 1>::Zero()),
      TGElement::Identity(), fixture::kTol));
  EXPECT(assert_equal((Vector)Eigen::Matrix<double, 18, 1>::Zero(),
                      (Vector)TGElement::Identity().Logmap(), fixture::kTol));

  const Eigen::Matrix<double, 18, 1> v = fixture::makeDelta();
  EXPECT(assert_equal((Vector)v, (Vector)TGElement::Expmap(v).Logmap(),
                      fixture::kTolL));
}

// Exp intertwines the adjoint: Exp(Ad_X w) = X Exp(w) X^-1. This holds only for
// the exact exponential, so it checks that the Xi series has converged.
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
}

// dexp is not implemented; requesting it must throw rather than hand back an
// uninitialized Jacobian.
TEST(TGElement, ExpmapLogmapThrowOnJacobianRequest) {
  using Tr = traits<TGElement>;
  Eigen::Matrix<double, 18, 18> H;
  const Eigen::Matrix<double, 18, 1> v = Eigen::Matrix<double, 18, 1>::Zero();
  CHECK_EXCEPTION(Tr::Expmap(v, H), std::runtime_error);
  CHECK_EXCEPTION(Tr::Logmap(TGElement::Identity(), H), std::runtime_error);
}

}  // namespace exponential
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
