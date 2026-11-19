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

using gtsam::Rot3;
using gtsam::Vector3;

struct MekfState {
  Rot3 R;
  Vector3 v;
  Vector3 p;
  Vector3 b_gyro;
  Vector3 b_accel;
  Vector3 b_mag;

  static constexpr int dimension = 18;

  using TangentVector = Eigen::Matrix<double, 18, 1>;
  using Jacobian = Eigen::Matrix<double, 18, 18>;

  MekfState();
  MekfState(const Rot3& R, const Vector3& v, const Vector3& p,
            const Vector3& b_gyro, const Vector3& b_accel,
            const Vector3& b_mag = Vector3::Zero());

  static MekfState identity();
};

}  // namespace mekf

// GTSAM Manifold traits for ManifoldEKF<MekfState>.
namespace gtsam {

template <>
struct traits<mekf::MekfState> {
  using ManifoldType = mekf::MekfState;
  using TangentVector = mekf::MekfState::TangentVector;
  using Jacobian = mekf::MekfState::Jacobian;
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
