#include <gtsam_unstable/tfg_inekf/Group.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/SO3.h>
#include <iostream>

// ============================================================================
//  TwoFrameGroup implementation.
//
//  Key structural fact (Sec. 5.3): at the level of the GROUP PRODUCT the
//  two-frames group is  G_TF = SO(3) |x R^12, where the 12-vector
//  u = (v, p, gamma_w, gamma_a) and every R^3 block transforms identically by
//  pure rotation:   u_{XY} = u_X + A_X u_Y.
//  The SE_2(3)-style coupling (p^ A, v^ A in the adjoint; left Jacobian J_l in
//  Exp/Log) is shared by ALL four blocks. This makes every operation a single
//  loop over the four vector blocks. Exp/Log are EXACT (no fiber truncation).
// ============================================================================

namespace {

using gtsam::Matrix3;
using gtsam::Vector3;

inline Matrix3 skew(const Vector3& w) { return gtsam::skewSymmetric(w); }

// Left Jacobian of SO(3):  J_l(phi) = I + ((1-cos t)/t^2) W + ((t-sin t)/t^3) W^2
Matrix3 leftJacobian(const Vector3& phi) {
  const double t2 = phi.squaredNorm();
  const Matrix3 W = skew(phi);
  // t2 < 1e-8 (t < 1e-4) avoids catastrophic cancellation in 1-cos(t)
  if (t2 < 1e-8) return Matrix3::Identity() + 0.5 * W;  // 1st-order limit
  const double t = std::sqrt(t2);
  const double a = (1.0 - std::cos(t)) / t2;
  const double b = (t - std::sin(t)) / (t2 * t);
  return Matrix3::Identity() + a * W + b * (W * W);
}

// Inverse left Jacobian:  J_l^{-1}(phi) = I - 1/2 W + (1-alpha)/t^2 W^2,
// with alpha = (t/2) cot(t/2).
Matrix3 leftJacobianInverse(const Vector3& phi) {
  const double t2 = phi.squaredNorm();
  const Matrix3 W = skew(phi);
  // t2 < 1e-8 avoids catastrophic cancellation
  if (t2 < 1e-8) return Matrix3::Identity() - 0.5 * W;  // 1st-order limit
  const double t = std::sqrt(t2);
  const double half = 0.5 * t;
  const double alpha = half * std::cos(half) / std::sin(half);
  return Matrix3::Identity() - 0.5 * W + ((1.0 - alpha) / t2) * (W * W);
}

}  // namespace

namespace tfg {

// --- Construction -----------------------------------------------------------

TwoFrameGroup::TwoFrameGroup()
    : R(Rot3::Identity()),
      v(Vector3::Zero()),
      p(Vector3::Zero()),
      gamma_omega(Vector3::Zero()),
      gamma_accel(Vector3::Zero()) {}

TwoFrameGroup::TwoFrameGroup(const Rot3& R, const Vector3& v, const Vector3& p,
                             const Vector3& gamma_omega,
                             const Vector3& gamma_accel)
    : R(R), v(v), p(p), gamma_omega(gamma_omega), gamma_accel(gamma_accel) {}

TwoFrameGroup TwoFrameGroup::Identity() { return TwoFrameGroup(); }

// --- State <-> group coordinates (Eq. B.31:  b = A^{-1} * (-gamma)) ----------

TwoFrameGroup TwoFrameGroup::FromState(const Rot3& R, const Vector3& v,
                                       const Vector3& p, const Vector3& b_omega,
                                       const Vector3& b_accel) {
  // gamma = -A * b
  return TwoFrameGroup(R, v, p, -(R * b_omega), -(R * b_accel));
}

Eigen::Vector3d TwoFrameGroup::bias_omega() const {
  return R.unrotate(-gamma_omega);  // A^{-1} * (-gamma_w)
}
Eigen::Vector3d TwoFrameGroup::bias_accel() const {
  return R.unrotate(-gamma_accel);
}

// --- Group operations (Sec. 5.3) --------------------------------------------

// X*Y = ( C_X C_Y , gamma_X + A_X * gamma_Y ); every block: u_X + R_X u_Y.
TwoFrameGroup TwoFrameGroup::operator*(const TwoFrameGroup& Y) const {
  return TwoFrameGroup(R * Y.R,
                       v + R * Y.v,
                       p + R * Y.p,
                       gamma_omega + R * Y.gamma_omega,
                       gamma_accel + R * Y.gamma_accel);
}

// X^{-1} = ( C^{-1} , -A^T * gamma ); every block: -R^T u.
TwoFrameGroup TwoFrameGroup::inverse() const {
  const Rot3 Ri = R.inverse();
  return TwoFrameGroup(Ri, Ri * (-v), Ri * (-p), Ri * (-gamma_omega),
                       Ri * (-gamma_accel));
}

// --- Lie maps (exact; shared left Jacobian on all four vector blocks) --------

TwoFrameGroup TwoFrameGroup::Expmap(const TangentVector& xi) {
  const Vector3 theta = xi.segment<3>(0);
  const Matrix3 Jl = leftJacobian(theta);
  return TwoFrameGroup(Rot3::Expmap(theta),
                       Jl * xi.segment<3>(3),    // v
                       Jl * xi.segment<3>(6),    // p
                       Jl * xi.segment<3>(9),    // gamma_w
                       Jl * xi.segment<3>(12));  // gamma_a
}

TwoFrameGroup::TangentVector TwoFrameGroup::Logmap() const {
  const Vector3 theta = Rot3::Logmap(R);
  const Matrix3 Jli = leftJacobianInverse(theta);
  TangentVector xi;
  xi.segment<3>(0) = theta;
  xi.segment<3>(3) = Jli * v;
  xi.segment<3>(6) = Jli * p;
  xi.segment<3>(9) = Jli * gamma_omega;
  xi.segment<3>(12) = Jli * gamma_accel;
  return xi;
}

// Ad_{(A,u)}: rotation block A on the diagonal; each vector block adds the
// coupling column skew(u_i) * A under d_theta.
TwoFrameGroup::Jacobian TwoFrameGroup::AdjointMap() const {
  const Matrix3 A = R.matrix();
  Jacobian Ad = Jacobian::Zero();
  Ad.block<3, 3>(0, 0) = A;
  Ad.block<3, 3>(3, 3) = A;
  Ad.block<3, 3>(6, 6) = A;
  Ad.block<3, 3>(9, 9) = A;
  Ad.block<3, 3>(12, 12) = A;
  Ad.block<3, 3>(3, 0) = skew(v) * A;
  Ad.block<3, 3>(6, 0) = skew(p) * A;
  Ad.block<3, 3>(9, 0) = skew(gamma_omega) * A;
  Ad.block<3, 3>(12, 0) = skew(gamma_accel) * A;
  return Ad;
}

Eigen::Matrix<double, 5, 5> TwoFrameGroup::C_matrix() const {
  Eigen::Matrix<double, 5, 5> C = Eigen::Matrix<double, 5, 5>::Zero();
  C.block<3, 3>(0, 0) = R.matrix();
  C.block<3, 1>(0, 3) = v;
  C.block<3, 1>(0, 4) = p;
  C(3, 3) = 1.0;
  C(4, 4) = 1.0;
  return C;
}

}  // namespace tfg

// ============================================================================
//  GTSAM traits
// ============================================================================

namespace gtsam {

using G = tfg::TwoFrameGroup;
using T = traits<G>;

bool T::Equals(const G& X, const G& Y, double tol) {
  return X.R.equals(Y.R, tol) && (X.v - Y.v).norm() < tol &&
         (X.p - Y.p).norm() < tol &&
         (X.gamma_omega - Y.gamma_omega).norm() < tol &&
         (X.gamma_accel - Y.gamma_accel).norm() < tol;
}

void T::Print(const G& X, const std::string& str) {
  std::cout << str;
  X.R.print("  R: ");
  std::cout << "  v:      " << X.v.transpose() << "\n"
            << "  p:      " << X.p.transpose() << "\n"
            << "  gamma_w: " << X.gamma_omega.transpose() << "\n"
            << "  gamma_a: " << X.gamma_accel.transpose() << "\n";
}

G T::Identity() { return G::Identity(); }

G T::Compose(const G& X, const G& Y, ChartJacobian Hx, ChartJacobian Hy) {
  if (Hx) *Hx = Y.inverse().AdjointMap();  // d/dX = Ad_{Y^{-1}}
  if (Hy) *Hy = Jacobian::Identity();      // d/dY = I
  return X * Y;
}

G T::Between(const G& X, const G& Y, ChartJacobian Hx, ChartJacobian Hy) {
  const G result = X.inverse() * Y;
  if (Hx) *Hx = -result.inverse().AdjointMap();
  if (Hy) *Hy = Jacobian::Identity();
  return result;
}

G T::Inverse(const G& X, ChartJacobian H) {
  if (H) *H = -X.AdjointMap();
  return X.inverse();
}

// Exact right chart Jacobian. The group factors as SO(3) |x R^12 where every
// vector block u_i maps to J_l(theta) u_i in Exp -- structurally identical to
// the SE(3)/SE_2(3) translation block. So the 15x15 Expmap derivative reuses
// gtsam's so3::DexpFunctor exactly as Pose3::Expmap does: the SO(3) right
// Jacobian J_r on every diagonal 3x3 block, and the coupling R^T * d(J_l u_i)/
// d(theta) under the d_theta column of each vector block.
G T::Expmap(const TangentVector& xi, ChartJacobian H) {
  if (!H) return G::Expmap(xi);
  const Vector3 theta = xi.segment<3>(0);
  const gtsam::so3::DexpFunctor local(theta);
  const auto J = local.Jacobian();
  const Matrix3 Jr = J.right();
  const Matrix3 Rt = local.expmap().transpose();  // R^T

  H->setZero();
  H->block<3, 3>(0, 0) = Jr;
  for (int o : {3, 6, 9, 12}) {
    Matrix3 Hi;
    J.applyLeft(xi.segment<3>(o), &Hi);  // d(J_l(theta) u_i)/d(theta)
    H->block<3, 3>(o, o) = Jr;
    H->block<3, 3>(o, 0) = Rt * Hi;
  }
  return G::Expmap(xi);
}

T::TangentVector T::Logmap(const G& X, ChartJacobian H) {
  const TangentVector xi = X.Logmap();
  if (H) {  // inverse of the Expmap right Jacobian at xi
    Jacobian Hexp;
    Expmap(xi, Hexp);
    *H = Hexp.inverse();
  }
  return xi;
}

T::Jacobian T::AdjointMap(const G& X) { return X.AdjointMap(); }

// Right retraction / logarithm (matches gtsam LieGroup default).
G T::Retract(const G& X, const TangentVector& xi) { return X * G::Expmap(xi); }

T::TangentVector T::Local(const G& X, const G& Y) {
  return (X.inverse() * Y).Logmap();
}

}  // namespace gtsam
