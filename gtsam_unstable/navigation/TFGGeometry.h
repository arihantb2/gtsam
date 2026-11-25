/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file TFGGeometry.h
 * @brief The two-frames group SO(3) x R^15, the state space of the
 *        TFG-InEKF.
 * @author Arihant Lunawat
 *
 * The two-frames group G_TF = SO(3) x R^15 has dimension 18 and acts on
 * itself by right multiplication. Its elements hold the attitude, velocity,
 * position and the gyro, accelerometer and magnetometer biases, each bias b
 * stored in the global frame as gamma = -R b.
 */

#pragma once
#include <gtsam/base/Lie.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/dllexport.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tfg {

/// Lie algebra g_TF as an 18-vector, blocks [R v p gamma_w gamma_a gamma_m].
using Tangent = Eigen::Matrix<double, 18, 1>;

class GTSAM_UNSTABLE_EXPORT TwoFrameGroup {
 public:
  static constexpr int dim = 18;
  using TangentVector = Tangent;
  using Jacobian = Eigen::Matrix<double, dim, dim>;

  Rot3 R;
  Vector3 v;
  Vector3 p;
  Vector3 gamma_omega;
  Vector3 gamma_accel;
  Vector3 gamma_mag;

  TwoFrameGroup();

  TwoFrameGroup(const Rot3& R, const Vector3& v, const Vector3& p,
                const Vector3& gamma_omega, const Vector3& gamma_accel,
                const Vector3& gamma_mag = Vector3::Zero());

  /// Build from physical state (T, b); gamma = -R * b.
  static TwoFrameGroup FromState(const Rot3& R, const Vector3& v,
                                 const Vector3& p, const Vector3& b_omega,
                                 const Vector3& b_accel,
                                 const Vector3& b_mag = Vector3::Zero());

  /// Physical biases b = R^{-1} (-gamma).
  Vector3 biasOmega() const;
  Vector3 biasAccel() const;
  Vector3 biasMag() const;

  static TwoFrameGroup Identity();
  TwoFrameGroup operator*(const TwoFrameGroup& Y) const;
  TwoFrameGroup inverse() const;

  static TwoFrameGroup Expmap(const TangentVector& xi);
  TangentVector Logmap() const;

  /// Adjoint Ad_X (18x18) for InvariantEKF covariance propagation.
  Jacobian AdjointMap() const;

  /// 5x5 matrix realisation of the SE_2(3) navigation part.
  gtsam::Matrix5 navigationMatrix() const;
};

}  // namespace tfg

template <>
struct GTSAM_UNSTABLE_EXPORT traits<tfg::TwoFrameGroup> {
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

namespace tfg {

/// Right group action phi(X, xi) = xi * X. The InEKF operates directly on
/// TwoFrameGroup, so this is only needed for equivariance checks.
GTSAM_UNSTABLE_EXPORT TwoFrameGroup phi(const TwoFrameGroup& X,
                                        const TwoFrameGroup& xi);

}  // namespace tfg
}  // namespace gtsam
