#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

namespace gtsam {
namespace tgeqf {

State phi(const TGElement& X, const State& xi) {
  State result;

  result.R = xi.R * X.first.rotation();
  result.v = xi.R.rotate(X.first.x(0)) + xi.v;
  result.p = xi.R.rotate(X.first.x(1)) + xi.p;

  Vector9 b_diff;
  b_diff.segment<3>(0) = xi.b_w - X.second.segment<3>(0);
  b_diff.segment<3>(3) = xi.b_a - X.second.segment<3>(3);
  b_diff.segment<3>(6) = xi.b_v - X.second.segment<3>(6);
  const Vector9 b_new = X.first.inverse().AdjointMap() * b_diff;
  result.b_w = b_new.segment<3>(0);
  result.b_a = b_new.segment<3>(3);
  result.b_v = b_new.segment<3>(6);

  return result;
}

// Inverts phi block by block. The navigation part is immediate; the bias part
// follows from b_est = Ad_{A^{-1}}(b_ref - a), so a = b_ref - Ad_A(b_est).
TGElement phiInverse(const State& xi_ref, const State& xi_est) {
  const Rot3 R = xi_ref.R.inverse() * xi_est.R;
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = xi_ref.R.unrotate(xi_est.v - xi_ref.v);
  vp.col(1) = xi_ref.R.unrotate(xi_est.p - xi_ref.p);
  TGElement X(ExtendedPose3<2>(R, vp), Vector9::Zero());

  Vector9 b_est;
  b_est.segment<3>(0) = xi_est.b_w;
  b_est.segment<3>(3) = xi_est.b_a;
  b_est.segment<3>(6) = xi_est.b_v;
  const Vector9 Ad_b_est = X.first.AdjointMap() * b_est;

  Vector9 Xa;
  Xa.segment<3>(0) = xi_ref.b_w - Ad_b_est.segment<3>(0);
  Xa.segment<3>(3) = xi_ref.b_a - Ad_b_est.segment<3>(3);
  Xa.segment<3>(6) = xi_ref.b_v - Ad_b_est.segment<3>(6);
  X.second = Xa;
  return X;
}

TGSymmetry::Orbit::Orbit(const State& xi_ref) : xi_ref(xi_ref) {}

// phi(X, xi_ref) with optional Jacobian d(phi)/dX, in the origin chart
// theta = log_G . phi_xi_ref^-1. By freeness of phi this Jacobian is the
// identity at every X and every xi_ref, exactly.
State TGSymmetry::Orbit::operator()(const TGElement& X,
                                    Eigen::Matrix<double, 18, 18>* H) const {
  const State result = phi(X, xi_ref);

  if (H) {
    H->setIdentity();
  }

  return result;
}

TGSymmetry::Diffeomorphism::Diffeomorphism(const TGElement& X) : X(X) {}

// phi(X, xi) with optional Jacobian d(phi)/dxi, in the origin chart
// theta = log_G . phi_xi^-1. By the right-action axiom and freeness this
// Jacobian is the full group Adjoint Ad_{X^{-1}}, exactly.
State TGSymmetry::Diffeomorphism::operator()(
    const State& xi, Eigen::Matrix<double, 18, 18>* H) const {
  const State result = phi(X, xi);

  if (H) {
    *H = X.inverse().AdjointMap();
  }

  return result;
}

}  // namespace tgeqf
}  // namespace gtsam
