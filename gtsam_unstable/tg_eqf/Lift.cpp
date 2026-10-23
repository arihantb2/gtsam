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
    const State& xi, OptionalJacobian<18, 18> D_lift) const {
  const Eigen::Matrix3d Rt = xi.R.transpose();
  Vector9 Lambda1;
  Lambda1.segment<3>(0) = u.w - xi.b_w;
  Lambda1.segment<3>(3) = u.a - xi.b_a + Rt * u.g_vec;
  Lambda1.segment<3>(6) = u.v - xi.b_v + Rt * xi.v;

  Vector9 b;
  b.segment<3>(0) = xi.b_w;
  b.segment<3>(3) = xi.b_a;
  b.segment<3>(6) = xi.b_v;
  Vector9 tau;
  tau.segment<3>(0) = u.tau_w;
  tau.segment<3>(3) = u.tau_a;
  tau.segment<3>(6) = u.tau_v;

  const Matrix9 ad_b = ExtendedPose3<2>::adjointMap(b);
  const Vector9 Lambda2 = ad_b * Lambda1 - tau;

  Eigen::Matrix<double, 18, 1> result;
  result.head<9>() = Lambda1;
  result.tail<9>() = Lambda2;

  if (D_lift) {
    // Analytic Jacobian in the State chart
    // [dR(3), dv(3), dp(3), dbw(3), dba(3), dbv(3)]. db/d(eps_tau) =
    // ad_se23(b), db/d(eps_b) = -I_9. Lambda_2 = ad_b[Lambda_1] - tau, so
    // d(Lambda_2) = ad_b d(Lambda_1) - ad_{Lambda_1} db.
    const Matrix9& db_deps_tau = ad_b;
    const Matrix9 db_deps_b =
        -Matrix9::Identity();
    const Matrix9 ad_Lambda1 =
        ExtendedPose3<2>::adjointMap(Lambda1);

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
  Vector9 w;
  w.segment<3>(0) = u.w;
  w.segment<3>(3) = u.a;
  w.segment<3>(6) = u.v;
  Vector9 tau;
  tau.segment<3>(0) = u.tau_w;
  tau.segment<3>(3) = u.tau_a;
  tau.segment<3>(6) = u.tau_v;

  const Matrix9 Ad_A_inv_matrix =
      X.first.inverse().AdjointMap();
  const Vector9 w_minus_a = w - X.second;
  Vector9 Omega_A_inv = Vector9::Zero();
  Omega_A_inv.segment<3>(6) = X.first.rotation().unrotate(-X.first.x(0));
  const Vector9 w_out = Ad_A_inv_matrix * w_minus_a + Omega_A_inv;
  const Vector9 tau_out = Ad_A_inv_matrix * tau;

  Input result;
  result.w = w_out.segment<3>(0);
  result.a = w_out.segment<3>(3);
  result.v = w_out.segment<3>(6);
  result.tau_w = tau_out.segment<3>(0);
  result.tau_a = tau_out.segment<3>(3);
  result.tau_v = tau_out.segment<3>(6);
  result.g_vec = u.g_vec;
  return result;
}

}  // namespace tgeqf
}  // namespace gtsam
