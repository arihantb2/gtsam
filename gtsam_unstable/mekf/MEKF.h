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
/// gtsam::tgeqf::ImuNoise.
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
  using Covariance1 = Eigen::Matrix<double, 1, 1>;

  /// Default variance (m^2) given to the pseudo-measured horizontal axes of a
  /// depth update, large enough that the update leaves x and y untouched.
  static constexpr double kDefaultHorizontalVariance = 1e3;

  MultiplicativeEKF(const MekfState& X0, const Covariance& P0);

  /// Strapdown predict + finite-diff F; P <- F P F^T + Qc dt.
  /// @param g_vec  Nav-frame gravity vector; its sign fixes the frame
  ///               convention (Z-down/NED: +g on z).
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& g_vec, const Covariance& Qc, double dt);

  /// Same predict with diagonal Qc from ImuNoise PSDs.
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& g_vec, const ImuNoise& noise,
                 double dt);

  /// GNSS position update (h = p, world-frame R_pos).
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

  /// DVL update (h = R^T v, state-dependent H, body-frame R_dvl).
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  /**
   * Pressure-sensor depth update from the world-frame z position.
   *
   * The depth is stacked onto the estimated horizontal position to form the
   * pseudo-position p_tilde = [p_x, p_y, z_depth]^T, which is fed through the
   * position output with the horizontal variance inflated so only the vertical
   * direction is corrected. The same pseudo-measurement route is used in all
   * three filters so they share one API; see Sec. V-B of
   * tg_eqf/docs/EqF_design_for_DVL_Depth_aided_INS.pdf.
   *
   * @param z_depth              Measured world-frame z position.
   * @param R_depth              Variance of z_depth, as a 1x1 matrix.
   * @param horizontal_variance  Variance assigned to the pseudo-measured x and
   *                             y axes.
   */
  void update_depth(double z_depth, const Covariance1& R_depth,
                    double horizontal_variance = kDefaultHorizontalVariance);

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
