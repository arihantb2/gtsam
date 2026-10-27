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
  using Covariance = Eigen::Matrix<double, 18, 18>;
  using Covariance15 = Eigen::Matrix<double, 15, 15>;
  using Covariance3 = Eigen::Matrix<double, 3, 3>;
  using Covariance1 = Eigen::Matrix<double, 1, 1>;

  /// Default variance (m^2) given to the pseudo-measured horizontal axes of a
  /// depth update, large enough that the update leaves x and y untouched.
  static constexpr double kDefaultHorizontalVariance = 1e3;

  /// @param X0 initial group state (TwoFrameGroup::FromState for physical
  /// coords)
  /// @param P0 initial 18-dim tangent covariance.
  TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0);

  /// Block-diagonal initial covariance: the 15 navigation and IMU-bias blocks
  /// [R v p gamma_w gamma_a] from Sigma_nav, then the magnetometer bias. Like
  /// the IMU-bias blocks, Sigma_mag_bias is used as the gamma_m-chart
  /// covariance, which is exact to first order when the bias estimate is zero.
  static Covariance initialCovariance(const Covariance15& Sigma_nav,
                                      const Covariance3& Sigma_mag_bias);

  /// IMU predict with lifted tangent process noise Qc (scaled by dt).
  /// @param g_vec  Global-frame gravity vector; its sign fixes the frame
  ///               convention (Z-down/NED: +g on z).
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& g_vec, const Covariance& Qc, double dt);

  /// IMU predict; builds Qd = B diag(sigma^2) B^T dt from ImuNoise PSDs.
  void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& g_vec, const ImuNoise& noise,
                 double dt);

  /// Position update: h = R^T(pi - p).
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

  /// DVL update: h_d = R^T v; R_dvl in body frame.
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  /**
   * Pressure-sensor depth update from the world-frame z position.
   *
   * The scalar model h(xi) = e_3^T p is not equivariant, so the depth is
   * stacked onto the estimated horizontal position to form the pseudo-position
   * p_tilde = [p_x, p_y, z_depth]^T, which is fed through the position output
   * with the horizontal variance inflated so only the vertical direction is
   * corrected. Since the horizontal entries come from the estimate rather than
   * the true state, this update is a second-order approximation. See Sec. V-B
   * of tg_eqf/docs/EqF_design_for_DVL_Depth_aided_INS.pdf.
   *
   * @param z_depth              Measured world-frame z position.
   * @param R_depth              Variance of z_depth, as a 1x1 matrix.
   * @param horizontal_variance  Variance assigned to the pseudo-measured x and
   *                             y axes.
   */
  void update_depth(double z_depth, const Covariance1& R_depth,
                    double horizontal_variance = kDefaultHorizontalVariance);

  /**
   * Biased magnetometer update, h = R^T m + b_m = R^T (m - gamma_m).
   *
   * @param z_mag  Measured body-frame magnetic field.
   * @param m      Reference field in the global frame, in the same units.
   * @param R_mag  Body-frame covariance of z_mag, used directly.
   */
  void update_magnetometer(const Eigen::Vector3d& z_mag,
                           const Eigen::Vector3d& m, const Covariance3& R_mag);

  /// Error state eps = Local(estimate, X_true) in the group's local chart, i.e.
  /// the chart covariance() lives in. Block order R, v, p, bg, ba, bm.
  TangentVector errorStateVector(const TwoFrameGroup& X_true) const;

  // State accessors
  gtsam::Rot3 attitude() const;
  Eigen::Vector3d velocity() const;
  Eigen::Vector3d position() const;
  Eigen::Vector3d bias_gyro() const;
  Eigen::Vector3d bias_accel() const;
  Eigen::Vector3d bias_mag() const;
};

}  // namespace tfg
