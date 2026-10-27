#pragma once
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>

// Two-Frames Group G_TF = SO(3) x| R^15 (dim 18): attitude, velocity, position
// and the gyro, accelerometer and magnetometer biases, each bias b stored in
// the global frame as gamma = -R b.

namespace tfg {

using gtsam::Rot3;

/// Lie algebra g_TF as an 18-vector, blocks [R v p gamma_w gamma_a gamma_m].
using Tangent = Eigen::Matrix<double, 18, 1>;

class TwoFrameGroup {
 public:
  static constexpr int dim = 18;
  using TangentVector = Tangent;
  using Jacobian = Eigen::Matrix<double, dim, dim>;

  Rot3 R;
  Eigen::Vector3d v;
  Eigen::Vector3d p;
  Eigen::Vector3d gamma_omega;
  Eigen::Vector3d gamma_accel;
  Eigen::Vector3d gamma_mag;

  TwoFrameGroup();

  TwoFrameGroup(const Rot3& R, const Eigen::Vector3d& v,
                const Eigen::Vector3d& p, const Eigen::Vector3d& gamma_omega,
                const Eigen::Vector3d& gamma_accel,
                const Eigen::Vector3d& gamma_mag = Eigen::Vector3d::Zero());

  /// Build from physical state (T, b); gamma = -R * b.
  static TwoFrameGroup FromState(
      const Rot3& R, const Eigen::Vector3d& v, const Eigen::Vector3d& p,
      const Eigen::Vector3d& b_omega, const Eigen::Vector3d& b_accel,
      const Eigen::Vector3d& b_mag = Eigen::Vector3d::Zero());

  /// Physical biases b = R^{-1} (-gamma).
  Eigen::Vector3d bias_omega() const;
  Eigen::Vector3d bias_accel() const;
  Eigen::Vector3d bias_mag() const;

  static TwoFrameGroup Identity();
  TwoFrameGroup operator*(const TwoFrameGroup& Y) const;
  TwoFrameGroup inverse() const;

  static TwoFrameGroup Expmap(const TangentVector& xi);
  TangentVector Logmap() const;

  /// Adjoint Ad_X (18x18) for InvariantEKF covariance propagation.
  Jacobian AdjointMap() const;

  /// 5x5 matrix realisation of the SE_2(3) navigation part.
  Eigen::Matrix<double, 5, 5> C_matrix() const;
};

}  // namespace tfg

namespace gtsam {

template <>
struct traits<tfg::TwoFrameGroup> {
  using ManifoldType = tfg::TwoFrameGroup;
  using TangentVector = tfg::Tangent;
  using Jacobian = tfg::TwoFrameGroup::Jacobian;
  using ChartJacobian =
      OptionalJacobian<tfg::TwoFrameGroup::dim, tfg::TwoFrameGroup::dim>;
  using structure_category = lie_group_tag;
  using group_flavor = multiplicative_group_tag;

  static constexpr int dimension = tfg::TwoFrameGroup::dim;
  static int GetDimension(const tfg::TwoFrameGroup&) { return dimension; }

  // Testable
  static bool Equals(const tfg::TwoFrameGroup& X, const tfg::TwoFrameGroup& Y,
                     double tol = 1e-9);
  static void Print(const tfg::TwoFrameGroup& X, const std::string& s = "");

  // Group
  static tfg::TwoFrameGroup Identity();
  static tfg::TwoFrameGroup Compose(const tfg::TwoFrameGroup& X,
                                    const tfg::TwoFrameGroup& Y,
                                    ChartJacobian Hx = {},
                                    ChartJacobian Hy = {});
  static tfg::TwoFrameGroup Between(const tfg::TwoFrameGroup& X,
                                    const tfg::TwoFrameGroup& Y,
                                    ChartJacobian Hx = {},
                                    ChartJacobian Hy = {});
  static tfg::TwoFrameGroup Inverse(const tfg::TwoFrameGroup& X,
                                    ChartJacobian H = {});

  // Lie group
  static tfg::TwoFrameGroup Expmap(const TangentVector& xi,
                                   ChartJacobian H = {});
  static TangentVector Logmap(const tfg::TwoFrameGroup& X,
                              ChartJacobian H = {});
  static Jacobian AdjointMap(const tfg::TwoFrameGroup& X);

  // Manifold
  static tfg::TwoFrameGroup Retract(const tfg::TwoFrameGroup& X,
                                    const TangentVector& xi);
  static TangentVector Local(const tfg::TwoFrameGroup& X,
                             const tfg::TwoFrameGroup& Y);
};

template <>
struct traits<const tfg::TwoFrameGroup> : traits<tfg::TwoFrameGroup> {};

}  // namespace gtsam
