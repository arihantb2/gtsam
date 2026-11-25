#pragma once
#include <gtsam/navigation/InvariantEKF.h>
#include <gtsam_unstable/navigation/TFGGeometry.h>

#include <Eigen/Dense>

// TFG-InEKF: biased INS on G_TF via InvariantEKF<TwoFrameGroup>. The process
// model (lift) and measurement outputs live here; the group and its action
// are in TFGGeometry.h.

namespace tfg {

/// Lifted IMU input for InvariantEKF::predict (biases are in the state).
struct ImuInput {
  Vector3 omega;
  Vector3 accel;
  Vector3 tau_omega = Vector3::Zero();
  Vector3 tau_accel = Vector3::Zero();
  /// Random-walk driver of the magnetometer bias.
  Vector3 tau_mag = Vector3::Zero();

  /// Gravity vector in the global frame {G}. The filter is frame-agnostic: the
  /// caller's choice of sign is what fixes which way +z points (Z-down/NED
  /// gives +g on z, Z-up/ENU gives -g). Defaults to zero, i.e. no gravity.
  Vector3 g_vec = Vector3::Zero();
};

/// Continuous-time IMU noise PSDs, per-axis (diagonal, not isotropic).
/// mag_bias_rw is the random-walk PSD of the magnetometer bias, not an IMU
/// channel; it lives here so propagate() takes a single noise struct.
struct ImuNoise {
  Vector3 gyro = Vector3::Zero();
  Vector3 accel = Vector3::Zero();
  Vector3 gyro_rw = Vector3::Zero();
  Vector3 accel_rw = Vector3::Zero();
  Vector3 mag_bias_rw = Vector3::Zero();
};

/// Equivariant lift Lambda(X, u). Each bias block has the same form,
/// Lambda_gamma = b x (omega - b_omega) - tau, magnetometer bias included.
GTSAM_UNSTABLE_EXPORT Tangent lift(const TwoFrameGroup& X, const ImuInput& u);

/// Df = d lift(X * Expmap(eps), u) / d eps at eps = 0 (bias->navigation
/// coupling).
GTSAM_UNSTABLE_EXPORT TwoFrameGroup::Jacobian liftJacobian(
    const TwoFrameGroup& X, const ImuInput& u);

/// Qd = B diag(sigma^2) B^T dt; B maps (n_omega, n_accel, tau_w, tau_a, tau_m)
/// into lift. Non-zero blocks: B[theta,n_w]=I, B[v,n_a]=I, B[g_w,n_w]=b_w^,
/// B[g_w,tau_w]=-I, B[g_a,n_w]=b_a^, B[g_a,tau_a]=-I, B[g_m,n_w]=b_m^,
/// B[g_m,tau_m]=-I.
GTSAM_UNSTABLE_EXPORT TwoFrameGroup::Jacobian inputNoiseCov(
    const TwoFrameGroup& X, const ImuNoise& noise, double dt);

/// Group increment U = Expmap(lift(X,u) * dt) for InvariantEKF::predict.
GTSAM_UNSTABLE_EXPORT TwoFrameGroup increment(const TwoFrameGroup& X,
                                              const ImuInput& u, double dt);

/**
 * Exact transition Jacobian of the discrete step X_next = X * increment(X, u,
 * dt) in the right chart, F = d Local(X_next, X(eps)_next) / d eps at eps = 0.
 *
 * Perturbing X by Expmap(eps) moves both the left factor and the lift, so
 *   F = Ad_{U^-1} + J_r(lift(X, u) dt) Df dt,
 * with U the increment, J_r the right Jacobian of the group exponential, and
 * Df = liftJacobian(X, u). No truncation in dt beyond the step itself.
 */
GTSAM_UNSTABLE_EXPORT TwoFrameGroup::Jacobian transitionJacobian(
    const TwoFrameGroup& X, const ImuInput& u, double dt);

/// DVL body-frame velocity output (natively equivariant).
struct GTSAM_UNSTABLE_EXPORT BodyVelocityOutput {
  static constexpr int dim = 3;
  using Jacobian = Eigen::Matrix<double, 3, 18>;

  /// Predicted body-frame velocity h_d = R^T v.
  static Vector3 predict(const TwoFrameGroup& xi);

  /// Output action psi_d(X, y) = R_X^T y + R_X^T v_X; for equivariance checks.
  static Vector3 output_action(const TwoFrameGroup& X, const Vector3& y);

  /// Output Jacobian H = [ y_hat^  I  0  0  0  0 ] in the right-retract chart.
  /// d_v block is identity (velocity perturbed in rotated frame, not R^T).
  static Jacobian jacobian(const TwoFrameGroup& xi_hat);
};

/// Position output (equivariant body-frame residual).
struct GTSAM_UNSTABLE_EXPORT PositionOutput {
  static constexpr int dim = 3;
  using Jacobian = Eigen::Matrix<double, 3, 18>;

  /// Body-frame residual h = R^T (pi - p).
  static Vector3 predict(const TwoFrameGroup& xi, const Vector3& pi);

  /// Output action rho_X(y) = R_X^T (y - p_X); for equivariance checks.
  static Vector3 output_action(const TwoFrameGroup& X, const Vector3& y);

  /// First-order output Jacobian C0 = [ y_hat^  0  -I  0  0  0 ] for the
  /// chart xi(eps) = xi_hat * Expmap(eps).
  static Jacobian jacobian(const TwoFrameGroup& xi_hat, const Vector3& pi);
};

/// Biased magnetometer output (equivariant). With the hard-iron bias stored
/// as gamma_m = -R b_m, h = R^T m + b_m = R^T (m - gamma_m): a
/// known-landmark output with the reference field m in place of a landmark
/// and gamma_m in place of the position, the same form as PositionOutput.
struct GTSAM_UNSTABLE_EXPORT MagnetometerOutput {
  static constexpr int dim = 3;
  using Jacobian = Eigen::Matrix<double, 3, 18>;

  /// Predicted body-frame field h = R^T (m - gamma_m) = R^T m + b_m.
  static Vector3 predict(const TwoFrameGroup& xi, const Vector3& m);

  /// Output action rho_X(y) = R_X^T (y - gamma_m,X); for equivariance checks.
  static Vector3 output_action(const TwoFrameGroup& X, const Vector3& y);

  /// First-order output Jacobian C0 = [ y_hat^  0  0  0  0  -I ] for the
  /// chart xi(eps) = xi_hat * Expmap(eps), with y_hat = predict(xi_hat, m).
  static Jacobian jacobian(const TwoFrameGroup& xi_hat, const Vector3& m);
};

class GTSAM_UNSTABLE_EXPORT TfgInEKF
    : public gtsam::InvariantEKF<TwoFrameGroup> {
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
  void propagate(const Vector3& omega, const Vector3& accel,
                 const Vector3& g_vec, const Covariance& Qc, double dt);

  /// IMU predict; builds Qd = B diag(sigma^2) B^T dt from ImuNoise PSDs.
  void propagate(const Vector3& omega, const Vector3& accel,
                 const Vector3& g_vec, const ImuNoise& noise, double dt);

  /// Position update: h = R^T(pi - p).
  void update_position(const Vector3& pi, const Covariance3& R_pos);

  /// DVL update: h_d = R^T v; R_dvl in body frame.
  void update_dvl(const Vector3& z_dvl, const Covariance3& R_dvl);

  /**
   * Pressure-sensor depth update from the world-frame z position.
   *
   * The scalar model h(xi) = e_3^T p is not equivariant, so the depth is
   * stacked onto the estimated horizontal position to form the pseudo-position
   * p_tilde = [p_x, p_y, z_depth]^T, which is fed through the position output
   * with the horizontal variance inflated so only the vertical direction is
   * corrected. Since the horizontal entries come from the estimate rather than
   * the true state, this update is a second-order approximation. This is the
   * depth pseudo-measurement construction of E. R. Potokar, K. Norman, and
   * J. G. Mangelson, "Invariant Extended Kalman Filtering for Underwater
   * Navigation," IEEE Robotics and Automation Letters, vol. 6, no. 3, 2021,
   * doi:10.1109/LRA.2021.3085167.
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
  void update_magnetometer(const Vector3& z_mag, const Vector3& m,
                           const Covariance3& R_mag);

  /// Error state eps = Local(estimate, X_true) in the group's local chart, i.e.
  /// the chart covariance() lives in. Block order R, v, p, bg, ba, bm.
  TangentVector errorStateVector(const TwoFrameGroup& X_true) const;

  // State accessors
  gtsam::Rot3 attitude() const;
  Vector3 velocity() const;
  Vector3 position() const;
  Vector3 bias_gyro() const;
  Vector3 bias_accel() const;
  Vector3 bias_mag() const;

  /**
   * Covariance reset Jacobian for a correction eta applied as X * Expmap(eta).
   *
   * The posterior error e ~ N(eta, P) satisfies X_true = X * Expmap(e); after
   * the correction the error is e' = Local(X * Expmap(eta), X_true), whose
   * derivative at e = eta is the right Jacobian of the group exponential at
   * eta. This is the posterior-mean reset, the same convention as
   * TGEqF::resetMatrix.
   */
  static Jacobian resetJacobian(const TangentVector& eta);

 private:
  /// Kalman update in the current chart followed by the covariance reset
  /// P <- B P B^T with B = resetJacobian(eta).
  void updateWithReset(const gtsam::Vector& prediction, const gtsam::Matrix& H,
                       const gtsam::Vector& z, const gtsam::Matrix& R);
};

}  // namespace tfg
