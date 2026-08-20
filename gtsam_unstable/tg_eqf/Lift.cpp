#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/Lift.h>

namespace gtsam {
namespace tgeqf {

Eigen::Matrix<double, 21, 1> Input::vector() const {
  Eigen::Matrix<double, 21, 1> result;
  result.segment<3>(0) = w;
  result.segment<3>(3) = a;
  result.segment<3>(6) = v;
  result.segment<3>(9) = tau_w;
  result.segment<3>(12) = tau_a;
  result.segment<3>(15) = tau_v;
  result.segment<3>(18) = g_vec;
  return result;
}

Input Input::from_vector(const Eigen::Matrix<double, 21, 1>& v) {
  Input u;
  u.w = v.segment<3>(0);
  u.a = v.segment<3>(3);
  u.v = v.segment<3>(6);
  u.tau_w = v.segment<3>(9);
  u.tau_a = v.segment<3>(12);
  u.tau_v = v.segment<3>(15);
  u.g_vec = v.segment<3>(18);
  return u;
}

Lift::Lift(const Input& u) : u(u) {}

// Closed form (see Lift.h): expanding T^{-1}(G-N)T for T=[R,v,p] leaves only
// R^T acting on g (in the "a" slot) and on the state velocity v (in the "v"
// slot); the N/-N homogeneous-row coupling cancels exactly.
Eigen::Matrix<double, 18, 1> Lift::operator()(
    const State& xi, Eigen::Matrix<double, 18, 18>* D_lift) const {
  const Eigen::Matrix3d Rt = xi.R.transpose();
  const se2_3 Lambda1 = {u.w - xi.b_w, u.a - xi.b_a + Rt * u.g_vec,
                         u.v - xi.b_v + Rt * xi.v};

  const se2_3 b = {xi.b_w, xi.b_a, xi.b_v};
  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};
  const Eigen::Matrix<double, 9, 9> ad_b = detail::ad_se23(b);
  const Eigen::Matrix<double, 9, 1> Lambda2 =
      ad_b * Lambda1.vector() - tau.vector();

  Eigen::Matrix<double, 18, 1> result;
  result.head<9>() = Lambda1.vector();
  result.tail<9>() = Lambda2;

  if (D_lift) {
    // Analytic Jacobian in the State chart
    // [dR(3), dv(3), dp(3), dbw(3), dba(3), dbv(3)]. db/d(eps_tau) =
    // ad_se23(b), db/d(eps_b) = -I_9. Lambda_2 = ad_b[Lambda_1] - tau, so
    // d(Lambda_2) = ad_b d(Lambda_1) - ad_{Lambda_1} db.
    const Eigen::Matrix<double, 9, 9> db_deps_tau = detail::ad_se23(b);
    const Eigen::Matrix<double, 9, 9> db_deps_b =
        -Eigen::Matrix<double, 9, 9>::Identity();
    const Eigen::Matrix<double, 9, 9> ad_Lambda1 = detail::ad_se23(Lambda1);

    D_lift->setZero();
    D_lift->block<3, 3>(3, 0) = gtsam::skewSymmetric(Rt * u.g_vec);
    D_lift->block<3, 3>(6, 0) = gtsam::skewSymmetric(Rt * xi.v);
    D_lift->block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
    D_lift->block<9, 9>(0, 0) -= db_deps_tau;
    D_lift->block<9, 9>(0, 9) = -db_deps_b;

    D_lift->block<9, 18>(9, 0) = ad_b * D_lift->block<9, 18>(0, 0);
    D_lift->block<9, 9>(9, 0) -= ad_Lambda1 * db_deps_tau;
    D_lift->block<9, 9>(9, 9) -= ad_Lambda1 * db_deps_b;
  }

  return result;
}

InputOrbit::InputOrbit(const Input& u) : u(u) {}

Input InputOrbit::operator()(const TGElement& X) const {
  const se2_3 w = {u.w, u.a, u.v};
  const se2_3 tau = {u.tau_w, u.tau_a, u.tau_v};

  const se2_3 w_minus_a = {w.w - X.a.w, w.a - X.a.a, w.v - X.a.v};
  const se2_3 Ad_A_inv_w_minus_a = X.Ad_A_inv(w_minus_a);
  const se2_3 Omega_A_inv = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                             X.R.unrotate(-X.v)};
  const se2_3 w_out =
      se2_3::from_vector(Ad_A_inv_w_minus_a.vector() + Omega_A_inv.vector());
  const se2_3 tau_out = se2_3::from_vector(X.Ad_A_inv(tau).vector());

  Input result;
  result.w = w_out.w;
  result.a = w_out.a;
  result.v = w_out.v;
  result.tau_w = tau_out.w;
  result.tau_a = tau_out.a;
  result.tau_v = tau_out.v;
  result.g_vec = u.g_vec;
  return result;
}

}  // namespace tgeqf
}  // namespace gtsam
