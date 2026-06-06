#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <gtsam/geometry/Rot3.h>

namespace {

// 9x9 matrix representation of Ad_{A_X^{-1}} acting on se_2(3):
//   xi -> (R^T omega, R^T(eta - p x omega), R^T(alpha - v x omega))
//
//   Ad_{A^{-1}} = [  R^T         0    0  ]
//                 [ -R^T [p]^x   R^T  0  ]
//                 [ -R^T [v]^x   0    R^T ]
Eigen::Matrix<double, 9, 9> Ad9inv(const tgeqf::TGGroupElement& X) {
  const Eigen::Matrix3d Rt  = X.R_X.transpose();
  const Eigen::Matrix3d RtP = -Rt * gtsam::skewSymmetric(X.p_X);
  const Eigen::Matrix3d RtV = -Rt * gtsam::skewSymmetric(X.v_X);

  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero();
  A.block<3, 3>(0, 0) = Rt;
  A.block<3, 3>(3, 0) = RtP;
  A.block<3, 3>(3, 3) = Rt;
  A.block<3, 3>(6, 0) = RtV;
  A.block<3, 3>(6, 6) = Rt;
  return A;
}

// 9x9 adjoint (Lie bracket) matrix of a se_2(3) element:
//   ad_gamma(xi) = [gamma, xi]
//
//   ad_gamma = [ gw^x    0     0   ]
//              [ gv^x   gw^x   0   ]
//              [ ga^x    0    gw^x ]
Eigen::Matrix<double, 9, 9> ad9(const tgeqf::se2_3& gamma) {
  const Eigen::Matrix3d Gw = gtsam::skewSymmetric(gamma.omega);
  const Eigen::Matrix3d Gv = gtsam::skewSymmetric(gamma.v_tilde);
  const Eigen::Matrix3d Ga = gtsam::skewSymmetric(gamma.accel);

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
// Reference: Fornasier 2023 Lemma 4.1 / proposal Eq. (13)
// ---------------------------------------------------------------------------
TGState phi(const TGGroupElement& X, const TGState& xi) {
  TGState result;

  // Navigation: T_xi * A_X
  result.R = xi.R * X.R_X;
  result.p = xi.R.rotate(X.p_X) + xi.p;
  result.v = xi.R.rotate(X.v_X) + xi.v;

  // Bias: Ad_{A_X^{-1}}(b_xi - a_X)
  const se2_3 b_diff = {xi.b_omega - X.a.omega,
                        xi.b_v     - X.a.v_tilde,
                        xi.b_a     - X.a.accel};
  const se2_3 b_new  = X.Ad_A_inv(b_diff);
  result.b_omega = b_new.omega;
  result.b_v     = b_new.v_tilde;
  result.b_a     = b_new.accel;

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
//   [  0_3          |  R_out     |  0_3       |  0_{3x9}  ]   delta_p_out
//   [  0_3          |  0_3       |  R_out     |  0_{3x9}  ]   delta_v_out
//   [  ad9(b_out)   |  0_{9x3}   |  0_{9x3}   |  -I_9     ]   delta_b_out
//
// where R_out = (xi_ref.R * X.R_X).matrix(), and b_out = Ad_{A_X^{-1}}(b_xi - a_X).
TGState TGSymmetry::Orbit::operator()(const TGGroupElement& X,
                                      Eigen::Matrix<double, 18, 18>* H) const {
  const TGState result = phi(X, xi_ref);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R_out = result.R.matrix();

    // Navigation blocks
    H->block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // delta_R_out / delta_omega
    H->block<3, 3>(3, 3) = R_out;                        // delta_p_out / delta_eta
    H->block<3, 3>(6, 6) = R_out;                        // delta_v_out / delta_alpha

    // Bias blocks
    const se2_3 b_out = {result.b_omega, result.b_v, result.b_a};
    H->block<9, 9>(9, 0) = ad9(b_out);   // delta_b_out / delta_tau
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
// Diffeomorphism Jacobian J (18x18), columns = tangent of xi = [delta_R(3); delta_p(3); delta_v(3); delta_b(9)]:
//
//   [  R_X^T            |  0_3  |  0_3  |  0_{3x9}     ]   delta_R_out
//   [  -R_xi [p_X]^x   |  I_3  |  0_3  |  0_{3x9}     ]   delta_p_out
//   [  -R_xi [v_X]^x   |  0_3  |  I_3  |  0_{3x9}     ]   delta_v_out
//   [  0_{9x3}          |  0    |  0    |  Ad9inv(X)   ]   delta_b_out
TGState TGSymmetry::Diffeomorphism::operator()(const TGState& xi,
                                               Eigen::Matrix<double, 18, 18>* H) const {
  const TGState result = phi(X, xi);

  if (H) {
    H->setZero();

    const Eigen::Matrix3d R    = xi.R.matrix();
    const Eigen::Matrix3d Rxt  = X.R_X.transpose();

    // Rotation row: delta_R_out = R_X^T * delta_R
    H->block<3, 3>(0, 0) = Rxt;

    // Position row: delta_p_out = delta_p - R_xi * [p_X]^x * delta_R
    H->block<3, 3>(3, 0) = -R * gtsam::skewSymmetric(X.p_X);
    H->block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

    // Velocity row: delta_v_out = delta_v - R_xi * [v_X]^x * delta_R
    H->block<3, 3>(6, 0) = -R * gtsam::skewSymmetric(X.v_X);
    H->block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();

    // Bias row: delta_b_out = Ad_{C_X^{-1}} * delta_b
    H->block<9, 9>(9, 9) = Ad9inv(X);
  }

  return result;
}

}  // namespace tgeqf
