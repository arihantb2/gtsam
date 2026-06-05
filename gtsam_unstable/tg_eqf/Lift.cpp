#include <gtsam_unstable/tg_eqf/Lift.h>

#include <gtsam/geometry/Rot3.h>

namespace {

// 9x9 adjoint (Lie bracket) matrix of a se_2(3) element:
//   ad_gamma(xi) = [gamma, xi]
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

// Omega(C^{-1}) in Fornasier 2023: for X=(A,a,b) in SE_2(3), Omega(X)=(0,0,a).
// Evaluated at C^{-1}, the third se_2(3) slot carries C^{-1}'s position a.
tgeqf::se2_3 omegaAtCinv(const tgeqf::TGGroupElement& X) {
  return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          X.R_X.unrotate(-X.p_X)};
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
  v.segment<3>(6)  = a_tilde;
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
  u.a_tilde   = v.segment<3>(6);
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
  const se2_3 w = {u.omega, u.v_tilde, u.a_tilde};
  return w.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::B_matrix(const TGState& xi) const {
  const se2_3 b = {xi.b_omega, xi.b_v, xi.b_a};
  return b.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::N_matrix() const {
  Eigen::Matrix<double, 5, 5> N = Eigen::Matrix<double, 5, 5>::Zero();
  N(3, 4) = 1.0;
  return N;
}

Eigen::Matrix<double, 5, 5> Lift::G_matrix(const TGInput& u) const {
  const se2_3 g = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), u.g_vec};
  return g.wedge();
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
      W_matrix(u) - B_matrix(xi) + N_matrix() +
      Tinv * (G_matrix(u) - N_matrix()) * T;
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
          W_matrix(u) - B_matrix(xi_p) + N_matrix() +
          Tpinv * (G_matrix(u) - N_matrix()) * Tp;
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
  const se2_3 w = {u.omega, u.v_tilde, u.a_tilde};
  const se2_3 tau = {u.tau_omega, u.tau_v, u.tau_a};

  // w' = Ad_{C^{-1}}(w - a) + Omega(C^{-1})  (Fornasier 2023 Lemma 8, Eq. 20)
  const se2_3 w_minus_a = {w.omega - X.a.omega, w.v_tilde - X.a.v_tilde,
                           w.a_tilde - X.a.a_tilde};
  const se2_3 w_ad      = X.Ad_AT_inv(w_minus_a);
  const se2_3 omega_C   = omegaAtCinv(X);
  const se2_3 w_out     = {w_ad.omega, w_ad.v_tilde,
                           w_ad.a_tilde + omega_C.a_tilde};

  const se2_3 tau_out = X.Ad_AT_inv(tau);

  TGInput result;
  result.omega     = w_out.omega;
  result.v_tilde   = w_out.v_tilde;
  result.a_tilde   = w_out.a_tilde;
  result.tau_omega = tau_out.omega;
  result.tau_v     = tau_out.v_tilde;
  result.tau_a     = tau_out.a_tilde;
  result.g_vec     = u.g_vec;
  return result;
}

}  // namespace tgeqf
