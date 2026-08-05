#pragma once
#include <gtsam/navigation/ManifoldEKF.h>
#include <gtsam_unstable/mekf/BodyVelocityOutput.h>
#include <gtsam_unstable/mekf/Dynamics.h>
#include <gtsam_unstable/mekf/PositionOutput.h>
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

// MultiplicativeEKF: baseline MEKF for biased INS. See mekf/README.md.
// State-dependent F (finite-diff); naive position/DVL updates vs geometric
// filters.

namespace mekf {

/// Continuous-time IMU noise PSDs. Same convention as tfg::ImuNoise /
/// tgeqf::ImuNoise.
struct ImuNoise {
  double gyro = 0.0;
  double accel = 0.0;
  double gyro_rw = 0.0;
  double accel_rw = 0.0;
};

class MultiplicativeEKF : public gtsam::ManifoldEKF<MekfState> {
 public:
  using Base = gtsam::ManifoldEKF<MekfState>;
  using Covariance = Eigen::Matrix<double, 15, 15>;
  using Covariance3 = Eigen::Matrix<double, 3, 3>;

  MultiplicativeEKF(const MekfState& X0, const Covariance& P0);

  /// Strapdown predict + finite-diff F; P <- F P F^T + Qc dt.
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Covariance& Qc, double dt);

  /// Same predict with diagonal Qc from ImuNoise PSDs.
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const ImuNoise& noise, double dt);

  /// GNSS position update (h = p, world-frame R_pos).
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

  /// DVL update (h = R^T v, state-dependent H, body-frame R_dvl).
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  /// Error state eps = Local(estimate, X_true) in the filter's local chart
  /// (right/body multiplicative attitude, additive vectors), i.e. the chart
  /// covariance() lives in. Block order R, v, p, bg, ba.
  TangentVector errorStateVector(const MekfState& X_true) const;

  gtsam::Rot3 attitude() const;
  Eigen::Vector3d velocity() const;
  Eigen::Vector3d position() const;
  Eigen::Vector3d bias_gyro() const;
  Eigen::Vector3d bias_accel() const;
};

}  // namespace mekf
