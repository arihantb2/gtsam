#include <gtsam_unstable/tg_eqf/Lift.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

// ---------------------------------------------------------------------------
// TGInput
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 21, 1> TGInput::vector() const {
  Eigen::Matrix<double, 21, 1> result;
  result.segment<3>(0)  = w;
  result.segment<3>(3)  = a;
  result.segment<3>(6)  = v;
  result.segment<3>(9)  = tau_w;
  result.segment<3>(12) = tau_a;
  result.segment<3>(15) = tau_v;
  result.segment<3>(18) = g_vec;
  return result;
}

TGInput TGInput::from_vector(const Eigen::Matrix<double, 21, 1>& v) {
  TGInput u;
  u.w     = v.segment<3>(0);
  u.a     = v.segment<3>(3);
  u.v     = v.segment<3>(6);
  u.tau_w = v.segment<3>(9);
  u.tau_a = v.segment<3>(12);
  u.tau_v = v.segment<3>(15);
  u.g_vec = v.segment<3>(18);
  return u;
}

// ---------------------------------------------------------------------------
// Lift helper matrices (Fornasier 2023, Theorem 9)
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 5, 5> Lift::W_matrix(const TGInput& u) const {
  const se2_3 w = {u.w, u.a, u.v};
  return w.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::B_matrix(const TGState& xi) const {
  const se2_3 b = {xi.b_w, xi.b_a, xi.b_v};
  return b.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::G_matrix(const TGInput& u) const {
  const se2_3 g = {Eigen::Vector3d::Zero(), u.g_vec, Eigen::Vector3d::Zero()};
  return g.wedge();
}

Eigen::Matrix<double, 5, 5> Lift::N_matrix() const {
  Eigen::Matrix<double, 5, 5> N = Eigen::Matrix<double, 5, 5>::Zero();
  N(3, 4) = 1.0;
  return N;
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
      W_matrix(u) - B_matrix(xi) + N_matrix() + Tinv * (G_matrix(u) - N_matrix()) * T;
  const se2_3 Lambda1 = se2_3::vee(L1_mat);

  const se2_3 b = {xi.b_w, xi.b_a, xi.b_v};
  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};
  const Eigen::Matrix<double, 9, 1> Lambda2 =
      ad_se23(b) * Lambda1.vector() - tau.vector();

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
      const se2_3 bp = {xi_p.b_w, xi_p.b_a, xi_p.b_v};
      const Eigen::Matrix<double, 9, 1> Lambda2p =
          ad_se23(bp) * Lambda1p.vector() - tau.vector();

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
  const se2_3 w = {u.w, u.a, u.v};
  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};

  // w' = [Ad_{A^{-1}}(w - a) + Omega(A^{-1}), Ad_{A^{-1}}tau]
  const se2_3 w_minus_a = {w.w - X.a.w, w.a - X.a.a,
                           w.v - X.a.v};
  const se2_3 Ad_A_inv_w_minus_a = X.Ad_A_inv(w_minus_a);
  const se2_3 Omega_A_inv = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), X.R.unrotate(-X.v)};
  const se2_3 w_out = se2_3::from_vector(Ad_A_inv_w_minus_a.vector() + Omega_A_inv.vector());
  const se2_3 tau_out = se2_3::from_vector(X.Ad_A_inv(tau).vector());

  TGInput result;
  result.w     = w_out.w;
  result.a     = w_out.a;
  result.v     = w_out.v;
  result.tau_w = tau_out.w;
  result.tau_a = tau_out.a;
  result.tau_v = tau_out.v;
  result.g_vec = u.g_vec;
  return result;
}

}  // namespace tgeqf
