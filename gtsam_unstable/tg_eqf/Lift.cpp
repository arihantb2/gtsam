#include <gtsam_unstable/tg_eqf/Lift.h>

#include <gtsam/geometry/Rot3.h>

namespace {

// 9x9 adjoint (Lie bracket) matrix of a se_2(3) element:
//   ad_gamma(xi) = [gamma, xi]
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

// f_1^0(A^{-1}) in vee coordinates (Eq. 10): the velocity-drift matrix of the
// inverse SE_2(3) part places the velocity of A^{-1} in the position slot.
// v_{A^{-1}} = -R^T v_X, so vee = (0, -R^T v_X, 0).
tgeqf::se2_3 f1_A_inv(const tgeqf::TGGroupElement& X) {
  return {Eigen::Vector3d::Zero(), X.R_X.unrotate(-X.v_X),
          Eigen::Vector3d::Zero()};
}

}  // namespace

namespace tgeqf {

// ---------------------------------------------------------------------------
// TGInput
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 21, 1> TGInput::vector() const {
  Eigen::Matrix<double, 21, 1> v;
  v.segment<3>(0)  = omega;
  v.segment<3>(3)  = v_tilde;
  v.segment<3>(6)  = accel;
  v.segment<3>(9)  = tau_omega;
  v.segment<3>(12) = tau_v;
  v.segment<3>(15) = tau_a;
  v.segment<3>(18) = g_vec;
  return v;
}

TGInput TGInput::from_vector(const Eigen::Matrix<double, 21, 1>& v) {
  TGInput u;
  u.omega     = v.segment<3>(0);
  u.v_tilde   = v.segment<3>(3);
  u.accel     = v.segment<3>(6);
  u.tau_omega = v.segment<3>(9);
  u.tau_v     = v.segment<3>(12);
  u.tau_a     = v.segment<3>(15);
  u.g_vec     = v.segment<3>(18);
  return u;
}

// ---------------------------------------------------------------------------
// Lift helper matrices (Fornasier 2023, Theorem 9)
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 5, 5> Lift::W_matrix(const TGInput& u) const {
  const se2_3 w = {u.omega, u.v_tilde, u.accel};
  return w.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::B_matrix(const TGState& xi) const {
  const se2_3 b = {xi.b_omega, xi.b_v, xi.b_a};
  return b.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::G_matrix(const TGInput& u) const {
  const se2_3 g = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), u.g_vec};
  return g.wedge();
}

// f1(T) (Eq. 5): 5x5 matrix with the global velocity v in the position column.
// T^{-1} f1(T) then places R^T v in the position-rate slot, encoding dp = v.
// Identity: T^{-1} f1(T) = N - T^{-1} N T, so this term replaces the paper's
// +N + T^{-1}(G - N)T (Theorem 9, Eq. 21) with +T^{-1} G T + T^{-1} f1(T).
// Position column is column 3 (0-indexed) under gtsam's T = (R, p, v) order.
Eigen::Matrix<double, 5, 5> Lift::f1_matrix(const TGState& xi) const {
  Eigen::Matrix<double, 5, 5> F = Eigen::Matrix<double, 5, 5>::Zero();
  F.block<3, 1>(0, 3) = xi.v;
  return F;
}

// ---------------------------------------------------------------------------
// Lift
// ---------------------------------------------------------------------------

Lift::Lift(const TGInput& u) : u(u) {}

Eigen::Matrix<double, 18, 1> Lift::operator()(
    const TGState& xi, Eigen::Matrix<double, 18, 18>* D_lift) const {
  const Eigen::Matrix<double, 5, 5> T    = xi.T_matrix();
  const Eigen::Matrix<double, 5, 5> Tinv = T.inverse();

  const Eigen::Matrix<double, 5, 5> L1_mat =
      W_matrix(u) - B_matrix(xi) +
      Tinv * G_matrix(u) * T +
      Tinv * f1_matrix(xi);
  const se2_3 Lambda1 = se2_3::vee(L1_mat);

  const se2_3 b = {xi.b_omega, xi.b_v, xi.b_a};
  const se2_3 tau = {u.tau_omega, u.tau_v, u.tau_a};
  const Eigen::Matrix<double, 9, 1> Lambda2 =
      ad9(b) * Lambda1.vector() - tau.vector();

  Eigen::Matrix<double, 18, 1> result;
  result.head<9>() = Lambda1.vector();
  result.tail<9>() = Lambda2;

  if (D_lift) {
    static constexpr double h = 1e-7;
    for (int j = 0; j < 18; ++j) {
      Eigen::Matrix<double, 18, 1> e =
          Eigen::Matrix<double, 18, 1>::Zero();
      e(j) = h;
      const TGState xi_p = gtsam::traits<TGState>::Retract(xi, e);

      const Eigen::Matrix<double, 5, 5> Tp    = xi_p.T_matrix();
      const Eigen::Matrix<double, 5, 5> Tpinv = Tp.inverse();
      const Eigen::Matrix<double, 5, 5> L1p_mat =
          W_matrix(u) - B_matrix(xi_p) +
          Tpinv * G_matrix(u) * Tp +
          Tpinv * f1_matrix(xi_p);
      const se2_3 Lambda1p = se2_3::vee(L1p_mat);
      const se2_3 bp = {xi_p.b_omega, xi_p.b_v, xi_p.b_a};
      const Eigen::Matrix<double, 9, 1> Lambda2p =
          ad9(bp) * Lambda1p.vector() - tau.vector();

      Eigen::Matrix<double, 18, 1> result_p;
      result_p.head<9>() = Lambda1p.vector();
      result_p.tail<9>() = Lambda2p;
      D_lift->col(j) = (result_p - result) / h;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// InputOrbit
// ---------------------------------------------------------------------------

InputOrbit::InputOrbit(const TGInput& u) : u(u) {}

TGInput InputOrbit::operator()(const TGGroupElement& X) const {
  const se2_3 w = {u.omega, u.v_tilde, u.accel};
  const se2_3 tau = {u.tau_omega, u.tau_v, u.tau_a};

  // w' = Ad_{C^{-1}}(w - a) + f_1^0(C^{-1})  (Eq. 10)
  const se2_3 w_minus_a = {w.omega - X.a.omega, w.v_tilde - X.a.v_tilde,
                           w.accel - X.a.accel};
  const se2_3 w_ad      = X.Ad_A_inv(w_minus_a);
  const se2_3 f1_inv    = f1_A_inv(X);
  const se2_3 w_out     = {w_ad.omega, w_ad.v_tilde + f1_inv.v_tilde,
                           w_ad.accel};

  const se2_3 tau_out = X.Ad_A_inv(tau);

  TGInput result;
  result.omega     = w_out.omega;
  result.v_tilde   = w_out.v_tilde;
  result.accel   = w_out.accel;
  result.tau_omega = tau_out.omega;
  result.tau_v     = tau_out.v_tilde;
  result.tau_a     = tau_out.accel;
  result.g_vec     = u.g_vec;
  return result;
}

}  // namespace tgeqf
