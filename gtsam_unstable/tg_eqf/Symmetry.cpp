#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

// ---------------------------------------------------------------------------
// Standalone right group action
//
// phi(X, xi) = (T_xi * A_X,  Ad_{A_X^{-1}}(b_xi - a_X))
//
// where:
//   T_xi * A_X is the SE_2(3) product (R_xi*R_X, R_xi*p_X + p_xi, R_xi*v_X + v_xi)
//   b_xi - a_X is the component-wise difference in se_2(3)
//   Ad_{A_X^{-1}} is the adjoint of A_X inverse acting on se_2(3)
//
// Reference: Fornasier et al., arXiv:2309.03765, Lemma 7 (Eq. 19)
// ---------------------------------------------------------------------------
TGState phi(const TGGroupElement& X, const TGState& xi) {
  TGState result;

  // Navigation: T_xi * A_X
  result.R = xi.R * X.R;
  result.v = xi.R.rotate(X.v) + xi.v;
  result.p = xi.R.rotate(X.p) + xi.p;

  // Bias: Ad_{A_X^{-1}}(b_xi - a_X)
  const se2_3 b_diff = {xi.b_w - X.a.w,
                        xi.b_a     - X.a.a,
                        xi.b_v     - X.a.v};
  const se2_3 b_new  = X.Ad_A_inv(b_diff);
  result.b_w = b_new.w;
  result.b_a = b_new.a;
  result.b_v = b_new.v;

  return result;
}

// ---------------------------------------------------------------------------
// Orbit
// ---------------------------------------------------------------------------

TGSymmetry::Orbit::Orbit(const TGState& xi_ref) : xi_ref(xi_ref) {}

// phi(X, xi_ref) with optional Jacobian d(phi)/dX.
//
// Orbit Jacobian J (18x18), columns = tangent of X = [delta_tau(9); delta_sigma(9)]:
//
//   [  I_3          |  0_3       |  0_3       |  0_{3x9}  ]   delta_R_out
//   [  0_3          |  R_out     |  0_3       |  0_{3x9}  ]   delta_v_out
//   [  0_3          |  0_3       |  R_out     |  0_{3x9}  ]   delta_p_out
//   [  ad_se23(b_out) |  0_{9x3} |  0_{9x3}   |  -I_9     ]   delta_b_out
//
// where R_out = (xi_ref.R * X.R_X).matrix(), and b_out = Ad_{A_X^{-1}}(b_xi - a_X).
TGState TGSymmetry::Orbit::operator()(const TGGroupElement& X,
                                      Eigen::Matrix<double, 18, 18>* H) const {
  const TGState result = phi(X, xi_ref);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R_out = result.R.matrix();

    // Navigation blocks
    H->block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // delta_R_out / delta_R
    H->block<3, 3>(3, 3) = R_out;                        // delta_v_out / delta_v
    H->block<3, 3>(6, 6) = R_out;                        // delta_p_out / delta_p

    // Bias blocks
    const se2_3 b_out = {result.b_w, result.b_a, result.b_v};
    H->block<9, 9>(9, 0) = detail::ad_se23(b_out);  // delta_b_out / delta_tau
    H->block<9, 9>(9, 9) = -Eigen::Matrix<double, 9, 9>::Identity();  // / delta_sigma
  }

  return result;
}

// ---------------------------------------------------------------------------
// Diffeomorphism
// ---------------------------------------------------------------------------

TGSymmetry::Diffeomorphism::Diffeomorphism(const TGGroupElement& X) : X(X) {}

// phi(X, xi) with optional Jacobian d(phi)/dxi.
//
// Diffeomorphism Jacobian J (18x18), columns = tangent of xi = [delta_R(3); delta_v(3); delta_p(3); delta_b(9)]:
//
//   [  R_X^T            |  0_3  |  0_3  |  0_{3x9}     ]   delta_R_out
//   [  -R_xi [v_X]^x    |  I_3  |  0_3  |  0_{3x9}     ]   delta_v_out
//   [  -R_xi [p_X]^x    |  0_3  |  I_3  |  0_{3x9}     ]   delta_p_out
//   [  0_{9x3}          |  0    |  0    |  Ad_SE23_inv(X) ]   delta_b_out
TGState TGSymmetry::Diffeomorphism::operator()(const TGState& xi,
                                               Eigen::Matrix<double, 18, 18>* H) const {
  const TGState result = phi(X, xi);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R    = xi.R.matrix();
    const Eigen::Matrix3d Rxt  = X.R.transpose();

    *H = Eigen::Matrix<double, 18, 18>::Zero();

    // Rotation row: delta_R_out = R_X^T * delta_R
    H->block<3, 3>(0, 0) = Rxt;

    // Velocity row: delta_v_out = delta_v - R_xi * [v_X]^x * delta_R
    H->block<3, 3>(3, 0) = -R * gtsam::skewSymmetric(X.v);
    H->block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

    // Position row: delta_p_out = delta_p - R_xi * [p_X]^x * delta_R
    H->block<3, 3>(6, 0) = -R * gtsam::skewSymmetric(X.p);
    H->block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();

    // Bias row: delta_b_out = Ad_{C_X^{-1}} * delta_b
    H->block<9, 9>(9, 9) = detail::Ad_SE23_inv(X);
  }

  return result;
}

}  // namespace tgeqf
