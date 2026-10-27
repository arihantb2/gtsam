/**
 * @file  testTGMagnetometerOutput.cpp
 * @brief The biased magnetometer output h_m(xi) = R^T m + bm, its output
 *        action R_A^T (y - d), and the order of accuracy of C0 against C*.
 *
 * Layer above testTGSymmetry.cpp, alongside testTGBodyVelocityOutput.cpp: the
 * same equivariant measurement structure, with the hard-iron bias bm and the
 * group's d component in the roles of the velocity and its translation.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/TGEqF.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-9;
constexpr double kTolL = 1e-7;

using Tangent = State::TangentVector;

/// Reference field in the global frame (normalized units).
const Vector3 kField(0.4, 0.05, 0.9);

bool veq(const Vector3& a, const Vector3& b, double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

State makeXi() {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = Vector3(1.0, -2.0, 0.5);
  vp.col(1) = Vector3(0.3, 0.1, -0.4);
  State xi;
  xi.T = ExtendedPose3<2>(Rot3::Rz(0.3) * Rot3::Rx(0.1), vp);
  xi.b = (Vector9() << 0.01, -0.02, 0.03, -0.1, 0.05, 0.0, 0.0, 0.1, -0.05)
             .finished();
  xi.bm = Vector3(0.05, -0.08, 0.03);
  return xi;
}

TGMagElement makeElement(const Rot3& R, const Vector3& v, const Vector3& p,
                         const Vector9& a, const Vector3& d) {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = v;
  vp.col(1) = p;
  return TGMagElement(TGElement(ExtendedPose3<2>(R, vp), a), d);
}

TGMagElement makeX() {
  return makeElement(
      Rot3::Rz(0.4) * Rot3::Rx(0.2), Vector3(1.0, -2.0, 0.5),
      Vector3(0.3, 0.1, -0.4),
      (Vector9() << 0.1, -0.1, 0.05, -0.2, 0.3, 0.0, 0.0, 0.1, -0.1).finished(),
      Vector3(0.1, -0.2, 0.15));
}

TGMagElement makeY() {
  return makeElement(
      Rot3::Ry(0.3) * Rot3::Rz(-0.1), Vector3(-0.5, 1.0, 2.0),
      Vector3(0.2, -0.3, 0.1),
      (Vector9() << 0.05, 0.0, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.05)
          .finished(),
      Vector3(-0.05, 0.12, 0.2));
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_action {

// psi is a right action on the output space, the axiom that lets the filter
// carry a measurement from the estimate's body frame to the origin.
TEST(MagnetometerOutput, IsARightAction) {
  const Vector3 y(0.3, -0.1, 0.2);
  const TGMagElement X = fixture::makeX(), Y = fixture::makeY();

  EXPECT(fixture::veq(
      y, MagnetometerMeasurement::output_action(TGMagElement::Identity(), y)));
  EXPECT(fixture::veq(MagnetometerMeasurement::output_action(X * Y, y),
                      MagnetometerMeasurement::output_action(
                          Y, MagnetometerMeasurement::output_action(X, y)),
                      fixture::kTolL));
}

// inverse_output_action is psi_{X^-1}, which for a group action is psi_X^-1.
TEST(MagnetometerOutput, InverseActionIsTheActionOfTheInverse) {
  const Vector3 y(0.3, -0.1, 0.2);
  const TGMagElement X = fixture::makeX();

  EXPECT(fixture::veq(MagnetometerMeasurement::output_action(X.inverse(), y),
                      MagnetometerMeasurement::inverse_output_action(X, y)));
  EXPECT(fixture::veq(y, MagnetometerMeasurement::inverse_output_action(
                             X, MagnetometerMeasurement::output_action(X, y))));
}

// Output equivariance h_m(phi(X, xi)) = psi_X(h_m(xi)), including elements
// with d != 0 that move the bias.
TEST(MagnetometerOutput, IsEquivariant) {
  const State xi = fixture::makeXi();

  fixture::Tangent logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4, 0.1, -0.1, 0.05,
      -0.2, 0.3, 0.0, 0.0, 0.1, -0.1, 0.2, -0.3, 0.1;

  for (const TGMagElement& X : {TGMagElement::Identity(), fixture::makeX(),
                                TGMagElement::Expmap(logX)}) {
    EXPECT(fixture::veq(
        MagnetometerMeasurement::predict(phi(X, xi), fixture::kField),
        MagnetometerMeasurement::output_action(
            X, MagnetometerMeasurement::predict(xi, fixture::kField)),
        fixture::kTolL));
  }
}

// At the identity origin the pulled-back measurement is the bias-corrected
// field in the global frame, R_hat (z - bm_hat) (notes, eq. mag-transport),
// so the innovation compares it directly with the reference field.
TEST(MagnetometerOutput, TransportedMeasurementIsTheBiasCorrectedGlobalField) {
  const State xi_hat = fixture::makeXi();
  const TGMagElement X_hat = phiInverse(State::identity(), xi_hat);
  const Vector3 z(0.2, 0.6, 0.7);

  EXPECT(fixture::veq(xi_hat.T.rotation().rotate(z - xi_hat.bm),
                      MagnetometerMeasurement::inverse_output_action(X_hat, z),
                      fixture::kTolL));
}

}  // namespace output_action
/* ************************************************************************* */

/* ************************************************************************* */
namespace output_matrix {

// The reference output at the identity origin is the reference field itself.
TEST(MagnetometerOutput, ReferenceOutputIsTheField) {
  EXPECT(fixture::veq(
      fixture::kField,
      MagnetometerMeasurement::predict(State::identity(), fixture::kField)));
}

// C0 is the chart derivative of the output map at the reference state,
// including the -I_3 block the semidirect chart gives the bias.
TEST(MagnetometerOutput, C0MatchesNumerical) {
  const State xi_ref = fixture::makeXi();
  const Eigen::Matrix<double, 3, 21> H_num =
      numericalDerivative11<Vector3, State>(
          [](const State& x) {
            return MagnetometerMeasurement::predict(x, fixture::kField);
          },
          xi_ref);

  EXPECT(assert_equal(
      (Matrix)MagnetometerMeasurement::jacobian_C0(xi_ref, fixture::kField),
      (Matrix)H_num, 1e-5));
}

// At convergence the transported measurement is the origin output itself, so
// the C* midpoint collapses onto C0.
TEST(MagnetometerOutput, CstarMeetsC0AtConvergence) {
  const State xi_ref = fixture::makeXi();
  const TGMagElement g = fixture::makeX();
  const Vector3 z =
      MagnetometerMeasurement::predict(phi(g, xi_ref), fixture::kField);

  EXPECT(assert_equal(
      (Matrix)MagnetometerMeasurement::jacobian_C0(xi_ref, fixture::kField),
      (Matrix)MagnetometerMeasurement::jacobian_Cstar(xi_ref, g,
                                                      fixture::kField, z),
      fixture::kTol));
}

// The attitude block is m^ at convergence, whose null space is the field
// direction: the magnetometer cannot see rotation about m.
TEST(MagnetometerOutput, AttitudeBlockIsBlindToRotationAboutTheField) {
  const State xi_ref = State::identity();
  const TGMagElement g = fixture::makeX();
  const Vector3 z =
      MagnetometerMeasurement::predict(phi(g, xi_ref), fixture::kField);
  const Eigen::Matrix<double, 3, 21> Cstar =
      MagnetometerMeasurement::jacobian_Cstar(xi_ref, g, fixture::kField, z);

  EXPECT(fixture::veq(Vector3::Zero(), Cstar.leftCols<3>() * fixture::kField));
  EXPECT((Cstar.leftCols<3>() * Vector3::UnitZ()).norm() > 0.1);
}

}  // namespace output_matrix
/* ************************************************************************* */

/* ************************************************************************* */
namespace accuracy {

const State kXiRef = fixture::makeXi();
const TGMagElement kG = fixture::makeX();

/// Body-frame field the true state at eps reports to the filter.
Vector3 measurementAt(const fixture::Tangent& eps) {
  return MagnetometerMeasurement::predict(
      phi(kG, traits<State>::Retract(kXiRef, eps)), fixture::kField);
}

/// The residual TGEqF::update_magnetometer linearizes: the pulled-back
/// measurement against the reference output, as a function of the error.
Vector3 residual(const fixture::Tangent& eps) {
  return MagnetometerMeasurement::inverse_output_action(kG,
                                                        measurementAt(eps)) -
         MagnetometerMeasurement::predict(kXiRef, fixture::kField);
}

double errorC0(const fixture::Tangent& eps) {
  return (residual(eps) -
          MagnetometerMeasurement::jacobian_C0(kXiRef, fixture::kField) * eps)
      .norm();
}

double errorCstar(const fixture::Tangent& eps) {
  return (residual(eps) - MagnetometerMeasurement::jacobian_Cstar(
                              kXiRef, kG, fixture::kField, measurementAt(eps)) *
                              eps)
      .norm();
}

/// Ratio of the linearization error at half amplitude to that at full: ~1/4 for
/// a second-order matrix, ~1/8 for a third-order one.
double halvingRatio(double (*error)(const fixture::Tangent&),
                    const fixture::Tangent& d) {
  return error(0.1 * d) / error(0.2 * d);
}

// Halving the attitude error cuts C*'s linearization error by ~8 and C0's by
// ~4: the midpoint differential follows the residual to third order.
TEST(MagnetometerOutput, CstarIsThirdOrderInAttitudeError) {
  fixture::Tangent d = fixture::Tangent::Zero();
  d.segment<3>(0) = Vector3(0.6, -0.5, 0.62).normalized();

  EXPECT(errorC0(0.2 * d) > 1e-4);
  EXPECT(errorCstar(0.2 * d) > 1e-8);
  EXPECT(errorCstar(0.1 * d) < errorC0(0.1 * d));

  EXPECT(halvingRatio(errorC0, d) > 0.20);  // ~1/4
  EXPECT(halvingRatio(errorC0, d) < 0.35);
  EXPECT(halvingRatio(errorCstar, d) > 0.05);  // ~1/8
  EXPECT(halvingRatio(errorCstar, d) < 0.20);
}

// Third order survives a mixed attitude-bias error: the semidirect exponential
// moves the bias by J_r(eps_R) eps_bm, whose second-order term cancels the one
// the C* midpoint contributes, as the SE_2(3) chart does for the DVL.
TEST(MagnetometerOutput, CstarIsThirdOrderInMixedAttitudeBiasError) {
  fixture::Tangent d = fixture::Tangent::Zero();
  d.segment<3>(0) = Vector3(0.4, -0.3, 0.5);
  d.segment<3>(18) = Vector3(-0.5, 0.2, 0.35);
  d.normalize();

  EXPECT(errorCstar(0.2 * d) > 1e-8);
  EXPECT(halvingRatio(errorCstar, d) > 0.05);  // ~1/8
  EXPECT(halvingRatio(errorCstar, d) < 0.20);
  EXPECT(errorCstar(0.2 * d) < 0.5 * errorC0(0.2 * d));
}

}  // namespace accuracy
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
