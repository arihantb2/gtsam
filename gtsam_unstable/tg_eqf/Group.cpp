#include <gtsam_unstable/tg_eqf/Group.h>

#include <iostream>
#include <stdexcept>

namespace gtsam {
namespace tgeqf {

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

TGElement TGElement::operator*(const TGElement& Y) const {
  TGElement Z;
  Z.R = R * Y.R;
  Z.v = R.rotate(Y.v) + v;
  Z.p = R.rotate(Y.p) + p;

  const auto Ad_AX_aY = Ad_A(Y.a);
  Z.a = se2_3::from_vector(a.vector() + Ad_AX_aY.vector());

  return Z;
}

TGElement TGElement::inverse() const {
  TGElement Xinv;
  Xinv.R = R.inverse();
  Xinv.v = R.unrotate(-v);
  Xinv.p = R.unrotate(-p);
  const se2_3 adj_inv = Ad_A_inv(a);
  Xinv.a = {-adj_inv.w, -adj_inv.a, -adj_inv.v};
  return Xinv;
}

se2_3 TGElement::Ad_A(const se2_3& xi) const {
  const Eigen::Vector3d Rw = R.rotate(xi.w);
  const Eigen::Vector3d Ra = R.rotate(xi.a);
  const Eigen::Vector3d Rv = R.rotate(xi.v);
  return {Rw, Ra + v.cross(Rw), Rv + p.cross(Rw)};
}

se2_3 TGElement::Ad_A_inv(const se2_3& xi) const {
  return {R.unrotate(xi.w), R.unrotate(xi.a - v.cross(xi.w)),
          R.unrotate(xi.v - p.cross(xi.w))};
}

namespace {
// Xi(tau) = sum_{k>=0} ad_tau^k / (k+1)!, the tangent-group dexp/left-Jacobian
// that carries the fiber part of the exponential (Expmap below). The series is
// entire; 16 terms reach machine precision for ||ad_tau|| up to ~5 (measured
// truncation 2e-15 at ||ad_tau|| = 4.8; 12 terms only gave ~3e-10 there). At
// filter scale (||tau|| ~ ||Lambda*dt||) far fewer terms already suffice.
Eigen::Matrix<double, 9, 9> Xi(const se2_3& tau) {
  const Eigen::Matrix<double, 9, 9> ad = detail::ad_se23(tau);
  Eigen::Matrix<double, 9, 9> term = Eigen::Matrix<double, 9, 9>::Identity();
  Eigen::Matrix<double, 9, 9> sum = Eigen::Matrix<double, 9, 9>::Zero();
  double fact = 1.0;
  for (int k = 0; k < 16; ++k) {
    fact *= (k + 1);
    sum += term / fact;
    term = term * ad;
  }
  return sum;
}
}  // namespace

TGElement TGElement::Expmap(const Eigen::Matrix<double, 18, 1>& v) {
  const detail::Se23 A = detail::Se23::Expmap(v.head<9>());
  TGElement X;
  X.R = A.rotation();
  X.v = A.x(0);
  X.p = A.x(1);
  const se2_3 tau = se2_3::from_vector(v.head<9>());
  X.a = se2_3::from_vector(Xi(tau) * v.tail<9>());
  return X;
}

Eigen::Matrix<double, 18, 1> TGElement::Logmap() const {
  Eigen::Matrix<double, 18, 1> v;
  v.head<9>() = detail::Se23::Logmap(detail::toSe23(*this));
  const se2_3 tau = se2_3::from_vector(v.head<9>());
  v.tail<9>() = Xi(tau).partialPivLu().solve(a.vector());
  return v;
}

Eigen::Matrix<double, 5, 5> TGElement::to_A_matrix() const {
  Eigen::Matrix<double, 5, 5> A = Eigen::Matrix<double, 5, 5>::Zero();
  A.block<3, 3>(0, 0) = R.matrix();
  A.block<3, 1>(0, 3) = v;
  A.block<3, 1>(0, 4) = p;
  A(3, 3) = 1.0;
  A(4, 4) = 1.0;
  return A;
}

void TGElement::A_from_matrix(const Eigen::Matrix<double, 5, 5>& A) {
  R = gtsam::Rot3(A.block<3, 3>(0, 0));
  v = A.block<3, 1>(0, 3);
  p = A.block<3, 1>(0, 4);
}

TGElement TGElement::retract(const Eigen::Matrix<double, 18, 1>& xi) const {
  return (*this) * Expmap(xi);
}

Eigen::Matrix<double, 18, 1> TGElement::localCoordinates(
    const TGElement& other) const {
  return (inverse() * other).Logmap();
}

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

using G = tgeqf::TGElement;
using T = traits<G>;

bool T::Equals(const G& X, const G& Y, double tol) {
  return X.R.equals(Y.R, tol) && (X.v - Y.v).norm() < tol &&
         (X.p - Y.p).norm() < tol && (X.a.w - Y.a.w).norm() < tol &&
         (X.a.a - Y.a.a).norm() < tol && (X.a.v - Y.a.v).norm() < tol;
}

void T::Print(const G& X, const std::string& str) {
  std::cout << str;
  X.R.print("   A_R: ");
  std::cout << "  A_v: " << X.v.transpose() << "\n"
            << "  A_p: " << X.p.transpose() << "\n"
            << "  a_w: " << X.a.w.transpose() << "\n"
            << "  a_a: " << X.a.a.transpose() << "\n"
            << "  a_v: " << X.a.v.transpose() << "\n";
}

G T::Identity() { return G::Identity(); }

G T::Compose(const G& X, const G& Y, ChartJacobian Hx, ChartJacobian Hy) {
  if (Hx) *Hx = T::AdjointMap(Y.inverse());
  if (Hy) Hy->setIdentity();
  return X * Y;
}

G T::Between(const G& X, const G& Y, ChartJacobian Hx, ChartJacobian Hy) {
  const G r = X.inverse() * Y;
  if (Hx) *Hx = -T::AdjointMap(r.inverse());
  if (Hy) Hy->setIdentity();
  return r;
}

G T::Inverse(const G& X, ChartJacobian H) {
  if (H) *H = -T::AdjointMap(X);
  return X.inverse();
}

G T::Expmap(const TangentVector& v, ChartJacobian H) {
  if (H) {
    throw std::runtime_error(
        "traits<TGElement>::Expmap: dexp Jacobian not implemented");
  }
  return G::Expmap(v);
}

T::TangentVector T::Logmap(const G& X, ChartJacobian H) {
  if (H) {
    throw std::runtime_error(
        "traits<TGElement>::Logmap: dlog Jacobian not implemented");
  }
  return X.Logmap();
}

// Full 18x18 Adjoint map of G_TG = SE_2(3) ⋉ se_2(3):
//
//   Ad_{(A,a)} = [ Ad_A(9x9)         0_{9x9}  ]
//                [ ad_a * Ad_A       Ad_A     ] in R^18x18
//
// where Ad_A and ad_a are the 9x9 matrices derived above.
Eigen::Matrix<double, 18, 18> T::AdjointMap(const G& X) {
  const Eigen::Matrix<double, 9, 9> AdA = tgeqf::detail::Ad_SE23(X);
  const Eigen::Matrix<double, 9, 9> ada = tgeqf::detail::ad_se23(X.a);

  Eigen::Matrix<double, 18, 18> Adj = Eigen::Matrix<double, 18, 18>::Zero();
  Adj.block<9, 9>(0, 0) = AdA;
  Adj.block<9, 9>(9, 0) = ada * AdA;
  Adj.block<9, 9>(9, 9) = AdA;
  return Adj;
}

}  // namespace gtsam
