#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

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

TGSymmetry::Orbit::Orbit(const State& xi_ref) : xi_ref(xi_ref) {}

// phi(X, xi_ref) with optional Jacobian d(phi)/dX. Columns = tangent of X =
// [delta_tau(9); delta_sigma(9)]:
//
//   [  I_3            |  0_3    |  0_3    |  0_{3x9}  ]   delta_R_out
//   [  0_3            |  R_out  |  0_3    |  0_{3x9}  ]   delta_v_out
//   [  0_3            |  0_3    |  R_out  |  0_{3x9}  ]   delta_p_out
//   [  ad_se23(b_out) |  0_{9x3}|  0_{9x3}|  -I_9     ]   delta_b_out
//
// R_out = (xi_ref.R * X.R).matrix(), b_out = Ad_{A_X^{-1}}(b_xi - a_X).
State TGSymmetry::Orbit::operator()(const TGElement& X,
                                    Eigen::Matrix<double, 18, 18>* H) const {
  const State result = phi(X, xi_ref);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R_out = result.R.matrix();

    H->block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H->block<3, 3>(3, 3) = R_out;
    H->block<3, 3>(6, 6) = R_out;

    const se2_3 b_out = {result.b_w, result.b_a, result.b_v};
    H->block<9, 9>(9, 0) = detail::ad_se23(b_out);
    H->block<9, 9>(9, 9) = -Eigen::Matrix<double, 9, 9>::Identity();
  }

  return result;
}

TGSymmetry::Diffeomorphism::Diffeomorphism(const TGElement& X) : X(X) {}

// phi(X, xi) with optional Jacobian d(phi)/dxi. Columns = tangent of xi =
// [delta_R(3); delta_v(3); delta_p(3); delta_b(9)]:
//
//   [  R_X^T         |  0_3  |  0_3  |  0_{3x9}        ]   delta_R_out
//   [  -R_xi [v_X]^x |  I_3  |  0_3  |  0_{3x9}        ]   delta_v_out
//   [  -R_xi [p_X]^x |  0_3  |  I_3  |  0_{3x9}        ]   delta_p_out
//   [  0_{9x3}       |  0    |  0    |  Ad_SE23_inv(X) ]   delta_b_out
State TGSymmetry::Diffeomorphism::operator()(
    const State& xi, Eigen::Matrix<double, 18, 18>* H) const {
  const State result = phi(X, xi);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R = xi.R.matrix();
    const Eigen::Matrix3d Rxt = X.R.transpose();

    *H = Eigen::Matrix<double, 18, 18>::Zero();

    H->block<3, 3>(0, 0) = Rxt;

    H->block<3, 3>(3, 0) = -R * gtsam::skewSymmetric(X.v);
    H->block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

    H->block<3, 3>(6, 0) = -R * gtsam::skewSymmetric(X.p);
    H->block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();

    H->block<9, 9>(9, 9) = detail::Ad_SE23_inv(X);
  }

  return result;
}

}  // namespace tgeqf
