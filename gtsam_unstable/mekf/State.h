#pragma once
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>

// MekfState: xi = [R, v, p, b_gyro, b_accel, b_mag] on SO(3) x R^15 (dim 18).
// b_mag is the body-frame hard-iron bias of the magnetometer.
//
// Error convention (body/right multiplicative, additive on vectors):
//     R_true = R_hat * Exp(d_theta),  v = v_hat + d_v, ...
// Matches tg_eqf and tfg_inekf so NEES/error metrics are comparable.
// Tangent block order: see mekf/README.md.

namespace mekf {

struct MekfState {
  gtsam::Rot3 R;
  Eigen::Vector3d v;
  Eigen::Vector3d p;
  Eigen::Vector3d b_gyro;
  Eigen::Vector3d b_accel;
  Eigen::Vector3d b_mag;

  static constexpr int dimension = 18;

  MekfState();
  MekfState(const gtsam::Rot3& R, const Eigen::Vector3d& v,
            const Eigen::Vector3d& p, const Eigen::Vector3d& b_gyro,
            const Eigen::Vector3d& b_accel,
            const Eigen::Vector3d& b_mag = Eigen::Vector3d::Zero());

  static MekfState identity();
};

}  // namespace mekf

// GTSAM Manifold traits for ManifoldEKF<MekfState>.
namespace gtsam {

template <>
struct traits<mekf::MekfState> {
  using ManifoldType = mekf::MekfState;
  using TangentVector = Eigen::Matrix<double, 18, 1>;
  using Jacobian = Eigen::Matrix<double, 18, 18>;
  using structure_category = manifold_tag;

  static constexpr int dimension = mekf::MekfState::dimension;
  static int GetDimension(const mekf::MekfState&) { return dimension; }

  static mekf::MekfState Retract(const mekf::MekfState& xi,
                                 const TangentVector& delta);
  static TangentVector Local(const mekf::MekfState& xi_ref,
                             const mekf::MekfState& xi);

  static mekf::MekfState Identity() { return mekf::MekfState::identity(); }

  static bool Equals(const mekf::MekfState& a, const mekf::MekfState& b,
                     double tol = 1e-9);
  static void Print(const mekf::MekfState& xi, const std::string& s = "");
};

template <>
struct traits<const mekf::MekfState> : traits<mekf::MekfState> {};

}  // namespace gtsam
