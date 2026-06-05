#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol  = 1e-9;
static constexpr double kTolL = 1e-7;

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

static TGState makeXi() {
  TGState xi;
  xi.R       = Rot3::Rz(0.3) * Rot3::Rx(0.1);
  xi.p       = Eigen::Vector3d(1.0, -2.0, 0.5);
  xi.v       = Eigen::Vector3d(0.3, 0.1, -0.4);
  xi.b_omega = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi.b_v     = Eigen::Vector3d(-0.1, 0.05, 0.0);
  xi.b_a     = Eigen::Vector3d(0.0, 0.1, -0.05);
  return xi;
}

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

static TGInput makeU() {
  TGInput u;
  u.omega     = Eigen::Vector3d(0.2, -0.1, 0.05);
  u.v_tilde   = Eigen::Vector3d(-0.05, 0.1, 0.0);
  u.a_tilde   = Eigen::Vector3d(0.0, 0.0, 9.81);
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_v     = Eigen::Vector3d::Zero();
  u.tau_a     = Eigen::Vector3d::Zero();
  u.g_vec     = Eigen::Vector3d(0.0, 0.0, -9.81);
  return u;
}

static Eigen::Matrix<double, 18, 18> numericalLiftJacobian(
    const Lift& lift, const TGState& xi, double h = 1e-6) {
  Eigen::Matrix<double, 18, 18> J;
  const Eigen::Matrix<double, 18, 1> f0 = lift(xi);
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xip = traits<TGState>::Retract(xi, e);
    J.col(j) = (lift(xip) - f0) / h;
  }
  return J;
}

// ---------------------------------------------------------------------------
// TGInput
// ---------------------------------------------------------------------------

TEST(TGInput, VectorRoundTrip) {
  const TGInput u = makeU();
  const TGInput recovered = TGInput::from_vector(u.vector());
  EXPECT(veq(u.omega, recovered.omega));
  EXPECT(veq(u.v_tilde, recovered.v_tilde));
  EXPECT(veq(u.a_tilde, recovered.a_tilde));
  EXPECT(veq(u.tau_omega, recovered.tau_omega));
  EXPECT(veq(u.tau_v, recovered.tau_v));
  EXPECT(veq(u.tau_a, recovered.tau_a));
  EXPECT(veq(u.g_vec, recovered.g_vec));
}

// ---------------------------------------------------------------------------
// Lift value at origin
// ---------------------------------------------------------------------------

TEST(Lift, AtIdentityStateLambda1IsWPlusG) {
  const TGInput u = makeU();
  const TGState xi0 = TGState::identity();
  Lift lift(u);

  // At T=I, b=0: Lambda_1 = W + G (N cancels in (W-B+N) + (G-N)).
  const se2_3 w = {u.omega, u.v_tilde, u.a_tilde};
  const se2_3 g = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), u.g_vec};
  const se2_3 expected = se2_3::vee(w.wedge() + g.wedge());
  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi0);

  EXPECT(veq(expected.omega, Lambda.segment<3>(0), kTolL));
  EXPECT(veq(expected.v_tilde, Lambda.segment<3>(3), kTolL));
  EXPECT(veq(expected.a_tilde, Lambda.segment<3>(6), kTolL));
}

TEST(Lift, AtIdentityStateLambda2IsMinusTau) {
  TGInput u = makeU();
  u.tau_omega = Eigen::Vector3d(0.01, -0.02, 0.0);
  u.tau_v     = Eigen::Vector3d(0.0, 0.03, -0.01);
  u.tau_a     = Eigen::Vector3d(0.02, 0.0, 0.01);

  const TGState xi0 = TGState::identity();
  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi0);

  Eigen::Matrix<double, 9, 1> tau_vec;
  tau_vec.segment<3>(0) = u.tau_omega;
  tau_vec.segment<3>(3) = u.tau_v;
  tau_vec.segment<3>(6) = u.tau_a;
  EXPECT(assert_equal((Vector)(-tau_vec), (Vector)Lambda.tail<9>(), kTolL));
}

// Regression: the lift must encode the position kinematics dp = v. With the
// virtual velocity input set to the bias estimate (v_tilde = b_v, the filter's
// operating point), the position-rate slot of Lambda_1 must equal the body
// velocity R^T v. A constant-N lift drops this term and freezes position.
TEST(Lift, PositionRateEncodesBodyVelocity) {
  TGState xi = makeXi();
  TGInput u = makeU();
  u.v_tilde = xi.b_v;  // virtual velocity input cancels its bias

  const Eigen::Matrix<double, 18, 1> Lambda = Lift(u)(xi);
  const Eigen::Vector3d body_v = xi.R.unrotate(xi.v);
  EXPECT(veq(body_v, Lambda.segment<3>(3), kTolL));  // v_tilde slot == R^T v
}

// ---------------------------------------------------------------------------
// Lift equivariance
// ---------------------------------------------------------------------------

TEST(Lift, EquivarianceAtIdentity) {
  const TGInput u = makeU();
  const TGState xi = makeXi();
  const TGGroupElement I = TGGroupElement::Identity();
  Lift lift(u);

  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi);
  const Eigen::Matrix<double, 18, 1> lhs =
      Lift(InputOrbit(u)(I))(phi(I, xi));
  const Eigen::Matrix<double, 18, 1> rhs =
      traits<TGGroupElement>::AdjointMap(I.inverse()) * Lambda;

  EXPECT(assert_equal((Vector)lhs, (Vector)rhs, kTolL));
  EXPECT(assert_equal((Vector)lhs, (Vector)Lambda, kTolL));
}

static bool checkLiftEquivariance(const TGGroupElement& X, const TGState& xi,
                                  const TGInput& u) {
  Lift lift(u);
  const Eigen::Matrix<double, 18, 1> Lambda = lift(xi);
  const Eigen::Matrix<double, 18, 1> lhs =
      Lift(InputOrbit(u)(X))(phi(X, xi));
  const Eigen::Matrix<double, 18, 1> rhs =
      traits<TGGroupElement>::AdjointMap(X.inverse()) * Lambda;
  return assert_equal((Vector)lhs, (Vector)rhs, kTolL);
}

TEST(Lift, EquivarianceForGeneralX) {
  const TGInput u = makeU();
  const TGState xi = makeXi();
  EXPECT(checkLiftEquivariance(makeX(), xi, u));
}

TEST(Lift, EquivarianceForExpmapElements) {
  const TGInput u = makeU();
  const TGState xi = makeXi();

  Eigen::Matrix<double, 18, 1> logX;
  logX << 0.15, -0.1, 0.05, 1.0, -2.0, 0.5, 0.3, 0.1, -0.4,
         0.1, -0.1, 0.05, -0.2, 0.3, 0.0, 0.0, 0.1, -0.1;
  EXPECT(checkLiftEquivariance(TGGroupElement::Expmap(logX), xi, u));

  Eigen::Matrix<double, 18, 1> logY;
  logY << -0.2, 0.3, -0.1, -0.5, 1.2, -0.3, 0.0, 0.4, 0.2,
         0.05, 0.05, -0.05, 0.1, -0.1, 0.2, -0.1, 0.0, 0.15;
  EXPECT(checkLiftEquivariance(TGGroupElement::Expmap(logY), xi, u));
}

// ---------------------------------------------------------------------------
// Lift Jacobian
// ---------------------------------------------------------------------------

TEST(Lift, JacobianMatchesNumerical) {
  const TGInput u = makeU();
  const TGState xi = makeXi();
  Lift lift(u);

  Eigen::Matrix<double, 18, 18> H_anal;
  lift(xi, &H_anal);

  const Eigen::Matrix<double, 18, 18> H_num = numericalLiftJacobian(lift, xi);
  EXPECT(meq(H_anal, H_num, 1e-5));
}

// ---------------------------------------------------------------------------
// InputOrbit
// ---------------------------------------------------------------------------

TEST(InputOrbit, AtIdentityIsNoop) {
  const TGInput u = makeU();
  const TGInput result = InputOrbit(u)(TGGroupElement::Identity());
  EXPECT(veq(u.omega, result.omega));
  EXPECT(veq(u.v_tilde, result.v_tilde));
  EXPECT(veq(u.a_tilde, result.a_tilde));
  EXPECT(veq(u.tau_omega, result.tau_omega));
  EXPECT(veq(u.tau_v, result.tau_v));
  EXPECT(veq(u.tau_a, result.tau_a));
  EXPECT(veq(u.g_vec, result.g_vec));
}

TEST(InputOrbit, GravityUnchanged) {
  const TGInput u = makeU();
  const TGInput result = InputOrbit(u)(makeX());
  EXPECT(veq(u.g_vec, result.g_vec));
}

TEST(InputOrbit, TauTransformedByAdInv) {
  const TGInput u = makeU();
  const TGGroupElement X = makeX();
  const TGInput result = InputOrbit(u)(X);

  const se2_3 tau = {u.tau_omega, u.tau_v, u.tau_a};
  const se2_3 tau_expected = X.Ad_AT_inv(tau);
  EXPECT(veq(result.tau_omega, tau_expected.omega, kTolL));
  EXPECT(veq(result.tau_v, tau_expected.v_tilde, kTolL));
  EXPECT(veq(result.tau_a, tau_expected.a_tilde, kTolL));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
