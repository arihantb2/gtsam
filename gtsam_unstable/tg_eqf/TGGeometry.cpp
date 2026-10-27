#include <gtsam_unstable/tg_eqf/TGGeometry.h>

#include <iostream>

namespace gtsam {
namespace tgeqf {

// HX: X Exp(delta) has attitude R_A Exp(delta_R), so
// d(R_A d) = -R_A d^ delta_R.
Vector3 MagBiasAction::operator()(const TGElement& X, const Vector3& d,
                                  OptionalJacobian<3, 18> HX,
                                  OptionalJacobian<3, 3> Hd) const {
  const Matrix3 R_A = X.first.rotation().matrix();
  if (HX) {
    HX->setZero();
    HX->leftCols<3>() = -R_A * skewSymmetric(d);
  }
  if (Hd) *Hd = R_A;
  return R_A * d;
}

Matrix3 MagBiasAction::generator(const TGElement::TangentVector& u) {
  return skewSymmetric(u.head<3>());
}

// Chart: theta = log_G . phi_this^-1.
State State::retract(const State::TangentVector& delta) const {
  return phi(TGMagElement::Expmap(delta), *this);
}

State::TangentVector State::localCoordinates(const State& other) const {
  return TGMagElement::Logmap(phiInverse(*this, other));
}

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

bool traits<tgeqf::State>::Equals(const tgeqf::State& a, const tgeqf::State& b,
                                  double tol) {
  return a.T.equals(b.T, tol) && (a.b - b.b).norm() < tol &&
         (a.bm - b.bm).norm() < tol;
}

void traits<tgeqf::State>::Print(const tgeqf::State& xi,
                                 const std::string& str) {
  std::cout << str;
  xi.T.print("  T: ");
  std::cout << "  b: " << xi.b.transpose() << "\n";
  std::cout << "  bm: " << xi.bm.transpose() << "\n";
}

}  // namespace gtsam

namespace gtsam {
namespace tgeqf {

State phi(const TGMagElement& X, const State& xi) {
  const ExtendedPose3<2>& A = X.first.first;
  State result;
  result.T = xi.T * A;
  result.b = A.inverse().AdjointMap() * (xi.b - X.first.second);
  result.bm = A.rotation().unrotate(xi.bm - X.second);
  return result;
}

// Nav block is the ExtendedPose3 "between" of xi_ref.T and xi_est.T; bias
// blocks follow from b_est = Ad_{A^-1}(b_ref - a), so a = b_ref - Ad_A(b_est),
// and bm_est = R_A^T (bm_ref - d), so d = bm_ref - R_A bm_est.
TGMagElement phiInverse(const State& xi_ref, const State& xi_est) {
  const ExtendedPose3<2> A = xi_ref.T.inverse() * xi_est.T;
  const Vector9 a = xi_ref.b - A.AdjointMap() * xi_est.b;
  const Vector3 d = xi_ref.bm - A.rotation().rotate(xi_est.bm);
  return TGMagElement(TGElement(A, a), d);
}

TGSymmetry::Orbit::Orbit(const State& xi_ref) : xi_ref(xi_ref) {}

// d(phi)/dX is exactly I_21 everywhere, by freeness of phi.
State TGSymmetry::Orbit::operator()(const TGMagElement& X,
                                    State::Jacobian* H) const {
  const State result = phi(X, xi_ref);

  if (H) {
    H->setIdentity();
  }

  return result;
}

TGSymmetry::Diffeomorphism::Diffeomorphism(const TGMagElement& X) : X(X) {}

// d(phi)/dxi is exactly Ad_{X^-1}, by the right-action axiom and freeness.
State TGSymmetry::Diffeomorphism::operator()(const State& xi,
                                             State::Jacobian* H) const {
  const State result = phi(X, xi);

  if (H) {
    *H = X.inverse().AdjointMap();
  }

  return result;
}

}  // namespace tgeqf
}  // namespace gtsam
