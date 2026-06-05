#include <gtsam_unstable/tg_eqf/Group.h>

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/ExtendedPose3-inl.h>
#include <iostream>

namespace {

// SE_2(3) = ExtendedPose3<2>. Tangent vector is [omega; eta; alpha] in R^9.
using Se23 = gtsam::ExtendedPose3<2>;

Se23 toSe23(const tgeqf::TGGroupElement& X) {
  Eigen::Matrix<double, 3, 2> pv;
  pv.col(0) = X.p_X;
  pv.col(1) = X.v_X;
  return Se23(X.R_X, pv);
}

// 9x9 Adjoint of a SE_2(3) element on se_2(3):
// Ad_{(R,p,v)}(omega, eta, alpha) = (R*omega, R*eta + p×(R*omega), R*alpha + v×(R*omega))
//
//   Ad_C = [ R        0   0 ]
//           [ p^× R   R   0 ]
//           [ v^× R   0   R ]
Eigen::Matrix<double, 9, 9> Ad9(const tgeqf::TGGroupElement& X) {
  const Eigen::Matrix3d R  = X.R_X.matrix();
  const Eigen::Matrix3d pR = gtsam::skewSymmetric(X.p_X) * R;
  const Eigen::Matrix3d vR = gtsam::skewSymmetric(X.v_X) * R;

  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero();
  A.block<3, 3>(0, 0) = R;
  A.block<3, 3>(3, 0) = pR;
  A.block<3, 3>(3, 3) = R;
  A.block<3, 3>(6, 0) = vR;
  A.block<3, 3>(6, 6) = R;
  return A;
}

// 9x9 adjoint representation of gamma in se_2(3):
// ad_gamma(xi) = [gamma, xi] = (gamma_w x xi_w,
//                                gamma_w x xi_v + gamma_v x xi_w,
//                                gamma_w x xi_a + gamma_a x xi_w)
//
//   ad_gamma = [ gw^×    0     0   ]
//              [ gv^×   gw^×   0   ]
//              [ ga^×    0    gw^× ]
Eigen::Matrix<double, 9, 9> ad9(const tgeqf::se2_3& gamma) {
  const Eigen::Matrix3d Gw = gtsam::skewSymmetric(gamma.omega);
  const Eigen::Matrix3d Gv = gtsam::skewSymmetric(gamma.v_tilde);
  const Eigen::Matrix3d Ga = gtsam::skewSymmetric(gamma.a_tilde);

  Eigen::Matrix<double, 9, 9> M = Eigen::Matrix<double, 9, 9>::Zero();
  M.block<3, 3>(0, 0) = Gw;
  M.block<3, 3>(3, 0) = Gv;
  M.block<3, 3>(3, 3) = Gw;
  M.block<3, 3>(6, 0) = Ga;
  M.block<3, 3>(6, 6) = Gw;
  return M;
}

}  // namespace

namespace tgeqf {

// ============================================================
// se2_3
// ============================================================

Eigen::Matrix<double, 9, 1> se2_3::vector() const {
  Eigen::Matrix<double, 9, 1> v;
  v.segment<3>(0) = omega;
  v.segment<3>(3) = v_tilde;
  v.segment<3>(6) = a_tilde;
  return v;
}

se2_3 se2_3::from_vector(const Eigen::Matrix<double, 9, 1>& v) {
  return {v.segment<3>(0), v.segment<3>(3), v.segment<3>(6)};
}

Eigen::Matrix<double, 5, 5> se2_3::wedge() const {
  Eigen::Matrix<double, 5, 5> W = Eigen::Matrix<double, 5, 5>::Zero();
  W.block<3, 3>(0, 0) = gtsam::skewSymmetric(omega);
  W.block<3, 1>(0, 3) = v_tilde;
  W.block<3, 1>(0, 4) = a_tilde;
  return W;
}

se2_3 se2_3::vee(const Eigen::Matrix<double, 5, 5>& mat) {
  return {gtsam::Rot3::Vee(mat.block<3, 3>(0, 0)), mat.block<3, 1>(0, 3),
          mat.block<3, 1>(0, 4)};
}

// ============================================================
// TGGroupElement
// ============================================================

TGGroupElement TGGroupElement::Identity() {
  TGGroupElement X;
  X.R_X = gtsam::Rot3::Identity();
  X.p_X = Eigen::Vector3d::Zero();
  X.v_X = Eigen::Vector3d::Zero();
  X.a   = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero()};
  return X;
}

// XY = (C_X C_Y, a_X + Ad_{C_X}[a_Y])
TGGroupElement TGGroupElement::operator*(const TGGroupElement& Y) const {
  const Eigen::Vector3d Rw = R_X.rotate(Y.a.omega);
  TGGroupElement Z;
  Z.R_X = R_X * Y.R_X;
  Z.p_X = R_X.rotate(Y.p_X) + p_X;
  Z.v_X = R_X.rotate(Y.v_X) + v_X;
  Z.a = {a.omega  + Rw,
         a.v_tilde + R_X.rotate(Y.a.v_tilde) + p_X.cross(Rw),
         a.a_tilde + R_X.rotate(Y.a.a_tilde) + v_X.cross(Rw)};
  return Z;
}

// X^{-1} = (C^{-1}, -Ad_{C^{-1}}[a])
TGGroupElement TGGroupElement::inverse() const {
  TGGroupElement Xinv;
  Xinv.R_X = R_X.inverse();
  Xinv.p_X = R_X.unrotate(-p_X);
  Xinv.v_X = R_X.unrotate(-v_X);
  const se2_3 adj_inv = Ad_AT_inv(a);
  Xinv.a = {-adj_inv.omega, -adj_inv.v_tilde, -adj_inv.a_tilde};
  return Xinv;
}

// Ad_{C_X}[xi] = (R xi_w, R xi_v + p x (R xi_w), R xi_a + v x (R xi_w))
se2_3 TGGroupElement::Ad_AT(const se2_3& xi) const {
  const Eigen::Vector3d Rw = R_X.rotate(xi.omega);
  return {Rw,
          R_X.rotate(xi.v_tilde) + p_X.cross(Rw),
          R_X.rotate(xi.a_tilde) + v_X.cross(Rw)};
}

// Ad_{C_X^{-1}}[xi] = (R^T xi_w, R^T(xi_v - p x xi_w), R^T(xi_a - v x xi_w))
se2_3 TGGroupElement::Ad_AT_inv(const se2_3& xi) const {
  return {R_X.unrotate(xi.omega),
          R_X.unrotate(xi.v_tilde - p_X.cross(xi.omega)),
          R_X.unrotate(xi.a_tilde - v_X.cross(xi.omega))};
}

// Expmap([tau; sigma]):
//   SE_2(3) part: exact via ExtendedPose3<2>::Expmap(tau)
//   se_2(3) fiber: first-order Ξ ≈ I, so gamma = sigma
TGGroupElement TGGroupElement::Expmap(const Eigen::Matrix<double, 18, 1>& v) {
  const Se23 C = Se23::Expmap(v.head<9>());
  TGGroupElement X;
  X.R_X = C.rotation();
  X.p_X = C.x(0);
  X.v_X = C.x(1);
  X.a   = se2_3::from_vector(v.tail<9>());
  return X;
}

// Logmap(X):
//   tau   = Logmap_{SE_2(3)}(C)
//   sigma = a.vector()  (consistent with first-order Expmap)
Eigen::Matrix<double, 18, 1> TGGroupElement::Logmap() const {
  Eigen::Matrix<double, 18, 1> v;
  v.head<9>() = Se23::Logmap(toSe23(*this));
  v.tail<9>() = a.vector();
  return v;
}

Eigen::Matrix<double, 5, 5> TGGroupElement::AT_matrix() const {
  Eigen::Matrix<double, 5, 5> T = Eigen::Matrix<double, 5, 5>::Zero();
  T.block<3, 3>(0, 0) = R_X.matrix();
  T.block<3, 1>(0, 3) = p_X;
  T.block<3, 1>(0, 4) = v_X;
  T(3, 3) = 1.0;
  T(4, 4) = 1.0;
  return T;
}

}  // namespace tgeqf

// ============================================================
// gtsam traits
// ============================================================

namespace gtsam {

using G = tgeqf::TGGroupElement;
using T = traits<G>;

bool T::Equals(const G& X, const G& Y, double tol) {
  return X.R_X.equals(Y.R_X, tol) &&
         (X.p_X - Y.p_X).norm() < tol &&
         (X.v_X - Y.v_X).norm() < tol &&
         (X.a.omega   - Y.a.omega).norm()   < tol &&
         (X.a.v_tilde - Y.a.v_tilde).norm() < tol &&
         (X.a.a_tilde - Y.a.a_tilde).norm() < tol;
}

void T::Print(const G& X, const std::string& str) {
  std::cout << str;
  X.R_X.print("  R: ");
  std::cout << "  p: "     << X.p_X.transpose()       << "\n"
            << "  v: "     << X.v_X.transpose()       << "\n"
            << "  a_w: "   << X.a.omega.transpose()   << "\n"
            << "  a_v: "   << X.a.v_tilde.transpose() << "\n"
            << "  a_a: "   << X.a.a_tilde.transpose() << "\n";
}

G T::Identity() { return G::Identity(); }

G T::Compose(const G& X, const G& Y, ChartJacobian, ChartJacobian) {
  return X * Y;
}

G T::Between(const G& X, const G& Y, ChartJacobian, ChartJacobian) {
  return X.inverse() * Y;
}

G T::Inverse(const G& X, ChartJacobian) { return X.inverse(); }

G T::Expmap(const TangentVector& v, ChartJacobian) {
  return G::Expmap(v);
}

T::TangentVector T::Logmap(const G& X, ChartJacobian) { return X.Logmap(); }

// Full 18x18 Adjoint map of G_TG = SE_2(3) ⋉ se_2(3):
//
//   Ad_{(C,gamma)} = [ Ad_C(9x9)            0_{9x9}  ]
//                    [ ad_gamma * Ad_C       Ad_C     ]
//
// where Ad_C and ad_gamma are the 9x9 matrices derived above.
Eigen::Matrix<double, 18, 18> T::AdjointMap(const G& X) {
  const Eigen::Matrix<double, 9, 9> AC  = Ad9(X);
  const Eigen::Matrix<double, 9, 9> adg = ad9(X.a);

  Eigen::Matrix<double, 18, 18> Adj = Eigen::Matrix<double, 18, 18>::Zero();
  Adj.block<9, 9>(0, 0) = AC;
  Adj.block<9, 9>(9, 0) = adg * AC;
  Adj.block<9, 9>(9, 9) = AC;
  return Adj;
}

// Right retraction: X * Expmap(xi)
G T::Retract(const G& X, const TangentVector& xi) {
  return X * G::Expmap(xi);
}

// Right logarithm: Logmap(X^{-1} * Y)
T::TangentVector T::Local(const G& X, const G& Y) {
  return (X.inverse() * Y).Logmap();
}

}  // namespace gtsam
