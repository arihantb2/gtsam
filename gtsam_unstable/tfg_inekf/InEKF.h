#pragma once
#include <gtsam/navigation/InvariantEKF.h>

#include <Eigen/Dense>

#include "Group.h"
#include "Symmetry.h"

// TFG-IEKF: biased INS on G_TF via InvariantEKF<TwoFrameGroup>.

namespace tfg {

class TfgInEKF : public gtsam::InvariantEKF<TwoFrameGroup> {
 public:
  using Base = gtsam::InvariantEKF<TwoFrameGroup>;
  using Covariance = Eigen::Matrix<double, 15, 15>;
  using Covariance3 = Eigen::Matrix<double, 3, 3>;

  /// @param X0 initial group state (TwoFrameGroup::FromState for physical
  /// coords)
  /// @param P0 initial 15-dim tangent covariance.
  TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0);

  /// IMU predict with lifted tangent process noise Qc (scaled by dt).
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Covariance& Qc, double dt);

  /// IMU predict; builds Qd = B diag(sigma^2) B^T dt from ImuNoise PSDs.
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const ImuNoise& noise, double dt);

  /// Position update: h = R^T(pi - p).
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos,
                       bool use_cstar = false);

  /// DVL update: h_d = R^T v; R_dvl in body frame.
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  /// Error state eps = Local(estimate, X_true) in the group's local chart, i.e.
  /// the chart covariance() lives in. Block order R, v, p, bg, ba.
  TangentVector errorStateVector(const TwoFrameGroup& X_true) const;

  // State accessors
  gtsam::Rot3 attitude() const;
  Eigen::Vector3d velocity() const;
  Eigen::Vector3d position() const;
  Eigen::Vector3d bias_gyro() const;
  Eigen::Vector3d bias_accel() const;
};

}  // namespace tfg
