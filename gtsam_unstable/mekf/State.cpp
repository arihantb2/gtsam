#include <gtsam_unstable/mekf/State.h>

#include <iostream>

namespace mekf {

MekfState::MekfState()
    : v(Eigen::Vector3d::Zero()),
      p(Eigen::Vector3d::Zero()),
      b_gyro(Eigen::Vector3d::Zero()),
      b_accel(Eigen::Vector3d::Zero()) {}

MekfState::MekfState(const gtsam::Rot3& R, const Eigen::Vector3d& v,
                     const Eigen::Vector3d& p, const Eigen::Vector3d& b_gyro,
                     const Eigen::Vector3d& b_accel)
    : R(R), v(v), p(p), b_gyro(b_gyro), b_accel(b_accel) {}

MekfState MekfState::identity() { return MekfState(); }

} // namespace mekf

namespace gtsam {

// Retraction at xi. Body/right multiplicative on attitude, additive on vectors.
// delta order = [d_theta(3); d_v(3); d_p(3); d_b_gyro(3); d_b_accel(3)].
mekf::MekfState traits<mekf::MekfState>::Retract(const mekf::MekfState& xi,
                                                 const TangentVector& delta) {
    mekf::MekfState result;
    result.R       = xi.R * Rot3::Expmap(delta.segment<3>(0));
    result.v       = xi.v       + delta.segment<3>(3);
    result.p       = xi.p       + delta.segment<3>(6);
    result.b_gyro  = xi.b_gyro  + delta.segment<3>(9);
    result.b_accel = xi.b_accel + delta.segment<3>(12);
    return result;
}

// Local(xi_ref, xi): inverse of Retract.
// d_theta = Logmap(R_ref^T R); additive differences on the vector blocks.
traits<mekf::MekfState>::TangentVector traits<mekf::MekfState>::Local(
    const mekf::MekfState& xi_ref, const mekf::MekfState& xi) {
    TangentVector delta;
    delta.segment<3>(0)  = Rot3::Logmap(xi_ref.R.inverse() * xi.R);
    delta.segment<3>(3)  = xi.v       - xi_ref.v;
    delta.segment<3>(6)  = xi.p       - xi_ref.p;
    delta.segment<3>(9)  = xi.b_gyro  - xi_ref.b_gyro;
    delta.segment<3>(12) = xi.b_accel - xi_ref.b_accel;
    return delta;
}

bool traits<mekf::MekfState>::Equals(const mekf::MekfState& a,
                                     const mekf::MekfState& b, double tol) {
    return a.R.equals(b.R, tol) &&
           (a.v       - b.v).norm()       < tol &&
           (a.p       - b.p).norm()       < tol &&
           (a.b_gyro  - b.b_gyro).norm()  < tol &&
           (a.b_accel - b.b_accel).norm() < tol;
}

void traits<mekf::MekfState>::Print(const mekf::MekfState& xi,
                                    const std::string& s) {
    std::cout << s;
    xi.R.print("  R: ");
    std::cout << "  v:       " << xi.v.transpose()       << "\n"
              << "  p:       " << xi.p.transpose()       << "\n"
              << "  b_gyro:  " << xi.b_gyro.transpose()  << "\n"
              << "  b_accel: " << xi.b_accel.transpose() << "\n";
}

} // namespace gtsam
