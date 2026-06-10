#include <gtsam_unstable/tg_eqf/Group.h>

#include <iostream>

namespace tgeqf {

// ============================================================
// se2_3
// ============================================================

Eigen::Matrix<double, 9, 1> se2_3::vector() const {
  Eigen::Matrix<double, 9, 1> result;
  result.segment<3>(0) = w;
  result.segment<3>(3) = a;
  result.segment<3>(6) = v;
  return result;
}

se2_3 se2_3::from_vector(const Eigen::Matrix<double, 9, 1>& v) {
  return {v.segment<3>(0), v.segment<3>(3), v.segment<3>(6)};
}

Eigen::Matrix<double, 5, 5> se2_3::wedge() const {
  Eigen::Matrix<double, 5, 5> W = Eigen::Matrix<double, 5, 5>::Zero();
  W.block<3, 3>(0, 0) = gtsam::skewSymmetric(w);
  W.block<3, 1>(0, 3) = a;
  W.block<3, 1>(0, 4) = v;
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
  X.R = gtsam::Rot3::Identity();
  X.v = Eigen::Vector3d::Zero();
  X.p = Eigen::Vector3d::Zero();
  X.a = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
         Eigen::Vector3d::Zero()};
  return X;
}

// XY = (A_X A_Y, a_X + Ad_{A_X}[a_Y])
TGGroupElement TGGroupElement::operator*(const TGGroupElement& Y) const {
  TGGroupElement Z;

  const auto A_X = to_A_matrix();
  const auto A_Y = Y.to_A_matrix();
  Z.A_from_matrix(A_X * A_Y);

  const auto Ad_AX_aY = Ad_A(Y.a);
  Z.a = se2_3::from_vector(a.vector() + Ad_AX_aY.vector());

  return Z;
}

// X^{-1} = (A^{-1}, -Ad_{A^{-1}}[a])
TGGroupElement TGGroupElement::inverse() const {
  TGGroupElement Xinv;
  Xinv.R = R.inverse();
  Xinv.v = R.unrotate(-v);
  Xinv.p = R.unrotate(-p);
  const se2_3 adj_inv = Ad_A_inv(a);
  Xinv.a = {-adj_inv.w, -adj_inv.a, -adj_inv.v};
  return Xinv;
}

// Ad_{A_X}[xi] = (R xi_w, R xi_a + v x (R xi_w), R xi_v + p x (R xi_w))
se2_3 TGGroupElement::Ad_A(const se2_3& xi) const {
  const Eigen::Vector3d Rw = R.rotate(xi.w);
  const Eigen::Vector3d Ra = R.rotate(xi.a);
  const Eigen::Vector3d Rv = R.rotate(xi.v);
  return {Rw,
          Ra + v.cross(Rw),
          Rv + p.cross(Rw)};
}

// Ad_{A_X^{-1}}[xi] = (R^T xi_w, R^T(xi_a - v x xi_w), R^T(xi_v - p x xi_w))
se2_3 TGGroupElement::Ad_A_inv(const se2_3& xi) const {
  return {R.unrotate(xi.w),
          R.unrotate(xi.a - v.cross(xi.w)),
          R.unrotate(xi.v - p.cross(xi.w))};
}

// Expmap([tau; sigma]):
//   SE_2(3) part: exact via ExtendedPose3<2>::Expmap(tau)
//   se_2(3) part: first-order Ξ ≈ I, so a = sigma
TGGroupElement TGGroupElement::Expmap(const Eigen::Matrix<double, 18, 1>& v) {
  const detail::Se23 A = detail::Se23::Expmap(v.head<9>());
  TGGroupElement X;
  X.A_from_matrix(A.matrix());
  X.a   = se2_3::from_vector(v.tail<9>());
  return X;
}

// Logmap(X):
//   tau   = Logmap_{SE_2(3)}(A)
//   sigma = a.vector()  (consistent with first-order Expmap)
Eigen::Matrix<double, 18, 1> TGGroupElement::Logmap() const {
  Eigen::Matrix<double, 18, 1> v;
  v.head<9>() = detail::Se23::Logmap(detail::toSe23(*this));
  v.tail<9>() = a.vector();
  return v;
}

Eigen::Matrix<double, 5, 5> TGGroupElement::to_A_matrix() const {
  // A = [R, v, p; 0, 1, 0; 0, 0, 1]
  Eigen::Matrix<double, 5, 5> A = Eigen::Matrix<double, 5, 5>::Zero();
  A.block<3, 3>(0, 0) = R.matrix();
  A.block<3, 1>(0, 3) = v;
  A.block<3, 1>(0, 4) = p;
  A(3, 3) = 1.0;
  A(4, 4) = 1.0;
  return A;
}

void TGGroupElement::A_from_matrix(const Eigen::Matrix<double, 5, 5>& A) {
  R = gtsam::Rot3(A.block<3, 3>(0, 0));
  v = A.block<3, 1>(0, 3);
  p = A.block<3, 1>(0, 4);
}

}  // namespace tgeqf

// ============================================================
// gtsam traits
// ============================================================

namespace gtsam {

using G = tgeqf::TGGroupElement;
using T = traits<G>;

bool T::Equals(const G& X, const G& Y, double tol) {
  return X.R.equals(Y.R, tol) &&
         (X.v - Y.v).norm() < tol &&
         (X.p - Y.p).norm() < tol &&
         (X.a.w - Y.a.w).norm()   < tol &&
         (X.a.a - Y.a.a).norm() < tol &&
         (X.a.v - Y.a.v).norm() < tol;
}

void T::Print(const G& X, const std::string& str) {
  std::cout << str;
  X.R.print("   A_R: ");
  std::cout << "  A_v: "   << X.v.transpose()   << "\n"
            << "  A_p: "   << X.p.transpose()   << "\n"
            << "  a_w: "   << X.a.w.transpose() << "\n"
            << "  a_a: "   << X.a.a.transpose() << "\n"
            << "  a_v: "   << X.a.v.transpose() << "\n";
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
//   Ad_{(A,a)} = [ Ad_A(9x9)         0_{9x9}  ]
//                [ ad_a * Ad_A       Ad_A     ] in R^18x18
//
// where Ad_A and ad_a are the 9x9 matrices derived above.
Eigen::Matrix<double, 18, 18> T::AdjointMap(const G& X) {
  const Eigen::Matrix<double, 9, 9> AdA  = tgeqf::detail::Ad_SE23(X);
  const Eigen::Matrix<double, 9, 9> ada = tgeqf::detail::ad_se23(X.a);

  Eigen::Matrix<double, 18, 18> Adj = Eigen::Matrix<double, 18, 18>::Zero();
  Adj.block<9, 9>(0, 0) = AdA;
  Adj.block<9, 9>(9, 0) = ada * AdA;
  Adj.block<9, 9>(9, 9) = AdA;
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
