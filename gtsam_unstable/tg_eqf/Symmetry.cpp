#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

namespace gtsam {
namespace tgeqf {

State phi(const TGElement& X, const State& xi) {
  State result;

  result.R = xi.R * X.R;
  result.v = xi.R.rotate(X.v) + xi.v;
  result.p = xi.R.rotate(X.p) + xi.p;

  const se2_3 b_diff = {xi.b_w - X.a.w, xi.b_a - X.a.a, xi.b_v - X.a.v};
  const se2_3 b_new = X.Ad_A_inv(b_diff);
  result.b_w = b_new.w;
  result.b_a = b_new.a;
  result.b_v = b_new.v;

  return result;
}

// Inverts phi block by block. The navigation part is immediate; the bias part
// follows from b_est = Ad_{A^{-1}}(b_ref - a), so a = b_ref - Ad_A(b_est).
TGElement phiInverse(const State& xi_ref, const State& xi_est) {
  TGElement X;
  X.R = xi_ref.R.inverse() * xi_est.R;
  X.v = xi_ref.R.unrotate(xi_est.v - xi_ref.v);
  X.p = xi_ref.R.unrotate(xi_est.p - xi_ref.p);

  const se2_3 Ad_b_est = X.Ad_A({xi_est.b_w, xi_est.b_a, xi_est.b_v});
  X.a = {xi_ref.b_w - Ad_b_est.w, xi_ref.b_a - Ad_b_est.a,
         xi_ref.b_v - Ad_b_est.v};
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
    *H = gtsam::traits<TGElement>::AdjointMap(X.inverse());
  }

  return result;
}

}  // namespace tgeqf
}  // namespace gtsam
