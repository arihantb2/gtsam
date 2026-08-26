#pragma once
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/tg_eqf/State.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <Eigen/Dense>
#include <optional>

namespace gtsam {
namespace tgeqf {

/**
 * Continuous-time IMU noise power spectral densities, mapped through the
 * lift differential B into process noise (see inputNoiseCov()). Per-axis
 * (diagonal, not isotropic): each density is a Vector3 of per-channel PSDs,
 * matching an IMU's independently-characterized x/y/z axes.
 *
 * Has no gyro/accel channel for the virtual input nu, and no random-walk
 * channel for its bias b_v: nu is held at zero rather than driven by a noisy
 * measurement, and b_v's own process noise instead enters separately as
 * virtual_bias_Q_.
 */
struct ImuNoise {
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_rw = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_rw = Eigen::Vector3d::Zero();
};

/**
 * TG-EqF biased INS filter on G = SE_2(3) ⋉ se_2(3).
 *
 * The base class keeps every matrix in **error coordinates**, the tangent space
 * at the fixed reference state xi_ref, so every measurement here supplies an
 * output matrix C* built at that reference state and hands it to
 * updateWithReset().
 */
class TGEqF : public gtsam::EquivariantFilter<State, TGSymmetry> {
 public:
  using Base = gtsam::EquivariantFilter<State, TGSymmetry>;
  using Covariance18 = Eigen::Matrix<double, 18, 18>;
  using Covariance15 = Eigen::Matrix<double, 15, 15>;
  using Covariance3 = Eigen::Matrix<double, 3, 3>;
  using Covariance1 = Eigen::Matrix<double, 1, 1>;

  /// Default variance (m^2) given to the pseudo-measured horizontal axes of a
  /// depth update, large enough that the update leaves x and y untouched.
  static constexpr double kDefaultHorizontalVariance = 1e6;

  /// Initial standard deviation of the virtual bias b_v.
  static constexpr double kVirtualBiasInitialSigma = 1e-6;
  static constexpr double kBiasSigma = 1e-3;

  /**
   * Full initial covariance built from the physical-state one, ordered
   * [R, v, p, b_w, b_a]. The physical block is used as given; the virtual-bias
   * block is kVirtualBiasInitialSigma^2 * I and uncorrelated with the rest.
   *
   * The v and p entries are read in the State chart, which takes SE_2(3)
   * logarithm coordinates on the navigation block. They are global-frame
   * velocity and position variances only when the state they are attached to
   * has identity rotation. Otherwise they are resolved in that state's body
   * frame and couple to the rotation block.
   */
  static Covariance18 initialCovariance(const Covariance15& Sigma_physical);

  /**
   * Construct at the identity origin (xi_ref = identity). This is the
   * interface examples should use, matching TfgInEKF and MEKF.
   *
   * @param X0  Initial group estimate; state is X0.
   * @param P0  Initial covariance of X0, expressed in the tangent chart at X0.
   */
  explicit TGEqF(const TGElement& X0, const Covariance18& P0);

  /**
   * Construct at an arbitrary reference state xi_ref. Exposed for testing;
   * examples should use the identity-origin constructor above.
   *
   * @param xi_ref  Fixed origin state.
   * @param P0      Initial covariance of the initial estimate phi(X0, xi_ref),
   *                expressed in the tangent chart at that estimate. It is
   *                transported internally to the origin chart at xi_ref, so
   *                covariance() returns Sigma0 unchanged right after
   *                construction. The transport is the identity when X0 is the
   *                identity.
   * @param X0      Initial group estimate; state is phi(X0, xi_ref).
   */
  explicit TGEqF(const State& xi_ref, const Covariance18& P0,
                 const TGElement& X0 = TGElement::Identity());

  // Enable/disable covariance reset after each update.
  void set_reset_step(bool enable) { reset_step_ = enable; }

  void set_virtual_bias_anchor(
      bool enable, const std::optional<Covariance3>& R_vb = std::nullopt);

  /**
   * Filter-only process noise on the physical bias states, as continuous-time
   * PSDs (variance units), added to Qc's b_w and b_a blocks by propagate().
   */
  void set_bias_process_noise(double gyro_psd, double accel_psd);

  /**
   * Reset transport J(delta_xi, delta_x); P <- J P J^T. Exposed for testing.
   */
  MatrixM resetMatrix(const TangentVector& delta_xi,
                      const TangentVector& delta_x) const;

  /**
   * IMU propagation with origin-chart Qc injected directly (P += Qc*dt).
   * Use the ImuNoise overload for lift-mapped process noise instead.
   */
  void propagate(const Eigen::Vector3d& w_meas, const Eigen::Vector3d& a_meas,
                 const Eigen::Vector3d& g_vec, const Covariance18& Qc,
                 double dt);

  /**
   * IMU propagation with Qc_eff = B Sigma B^T from IMU PSDs through the
   * lift differential (origin chart).
   */
  void propagate(const Eigen::Vector3d& w_meas, const Eigen::Vector3d& a_meas,
                 const Eigen::Vector3d& g_vec, const ImuNoise& noise,
                 double dt);

  // Qc_eff = B Sigma B^T; discrete covariance is Qc_eff * dt. For testing.
  Covariance18 inputNoiseCov(const ImuNoise& noise) const;

  /// Fixed-origin input lift orbit_jacobian0_ * G; reduces to G itself since
  /// orbit_jacobian0_ is always I_18. For testing.
  MatrixM input_lift() const { return input_lift_; }

  // DVL body-velocity update via h(xi) = R^T v, using the equivariant C*.
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  // Position update via h'(xi) = R^T(pi - p), using the equivariant C*.
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

  /**
   * Pressure-sensor depth update from the world-frame z position using the
   * right error pseudo position output.
   *
   * @param z_depth              Measured world-frame z position.
   * @param R_depth              Variance of z_depth, as a 1x1 matrix.
   * @param horizontal_variance  Variance assigned to the pseudo-measured x and
   *                             y axes.
   */
  void update_depth(double z_depth, const Covariance1& R_depth,
                    double horizontal_variance = kDefaultHorizontalVariance);

  /**
   * Pressure-sensor depth update through the non-equivariant scalar model
   * h(xi) = e_3^T p (see DepthOutput.h).
   *
   * @param z_depth  Measured world-frame z position.
   * @param R_depth  Variance of z_depth, as a 1x1 matrix.
   */
  void update_depth_direct(double z_depth, const Covariance1& R_depth);

  // Virtual-bias anchor b_v = 0 pseudo-measurement.
  void update_virtual_bias(const Covariance3& R_vb);

  // Error state and vector in the fixed-origin chart.
  State errorState(const State& xi_true) const;
  TangentVector errorStateVector(const State& xi_true) const;

  // State accessors.
  gtsam::Rot3 attitude() const;
  Eigen::Vector3d velocity() const;
  Eigen::Vector3d position() const;
  Eigen::Vector3d bias_gyro() const;
  Eigen::Vector3d bias_accel() const;
  Eigen::Vector3d bias_vel() const;

 private:
  // Compute the fixed-origin caches (orbit_jacobian0_, innovation_lift_,
  // input_lift_) from the current referenceState(). Called on construction.
  void computeOriginCaches();

  /**
   * Measurement update followed by the covariance reset.
   */
  void updateWithReset(const Eigen::VectorXd& prediction,
                       const Eigen::MatrixXd& Cstar, const Eigen::VectorXd& z,
                       const Eigen::MatrixXd& R);

  bool anchor_virtual_bias_ = false;

  // Process noise and measurement noise for the virtual-bias state.
  Covariance3 virtual_bias_R_ = 1e-6 * Covariance3::Identity();
  Covariance3 virtual_bias_Q_ = 1e-6 * Covariance3::Identity();

  /**
   * Process noise on b_w and b_a. Off by default
   */
  Covariance3 gyro_bias_Q_ = Covariance3::Zero();
  Covariance3 accel_bias_Q_ = Covariance3::Zero();

  bool reset_step_ = true;

  // Fixed-origin caches for the current reference state (xi_ref_).
  Eigen::Matrix<double, 18, 18> innovation_lift_;
  Eigen::Matrix<double, 18, 18> orbit_jacobian0_;
  Eigen::Matrix<double, 18, 18> input_lift_;
};

}  // namespace tgeqf
}  // namespace gtsam
