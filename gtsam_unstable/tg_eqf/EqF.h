#pragma once
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/tg_eqf/State.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <Eigen/Dense>
#include <optional>

namespace gtsam {
namespace tgeqf {

// Continuous-time IMU noise PSDs; mapped through lift differential B
struct ImuNoise {
  double gyro = 0.0;
  double accel = 0.0;
  double gyro_rw = 0.0;
  double accel_rw = 0.0;
};

/**
 * TG-EqF biased INS filter on G = SE_2(3) ⋉ se_2(3).
 *
 * The base class keeps every matrix in **error coordinates**, the tangent space
 * at the fixed reference state xi_ref, so every measurement here supplies an
 * output matrix C* built at that reference state and hands it to
 * updateWithReset(). Each output owns its own C*: a midpoint form for position
 * and DVL, an exact one for the virtual bias.
 *
 * These matrices are derivatives at the reference state, not at the current
 * estimate. The two have the same shape and agree only when the group estimate
 * is the identity.
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
  static constexpr double kDefaultHorizontalVariance = 1e3;

  /// Initial standard deviation of the virtual bias b_v. b_v has no physical
  /// counterpart to be uncertain about: it starts at zero and the b_v = 0
  /// anchor holds it there, so it is initialized as effectively known rather
  /// than with a user-set bias sigma.
  static constexpr double kVirtualBiasInitialSigma = 1e-6;

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
   * @param xi_ref  Fixed origin state (typically identity).
   * @param Sigma0  Initial covariance of the initial estimate phi(X0, xi_ref),
   *                expressed in the tangent chart at that estimate. It is
   *                transported internally to the origin chart at xi_ref, so
   *                covariance() returns Sigma0 unchanged right after
   *                construction. The transport is the identity when X0 is the
   *                identity.
   * @param X0      Initial group estimate; state is phi(X0, xi_ref).
   */
  explicit TGEqF(const State& xi_ref, const Covariance18& Sigma0,
                 const TGElement& X0 = TGElement::Identity());

  // Enable/disable covariance reset after each update.
  void set_reset_step(bool enable) { reset_step_ = enable; }

  void set_virtual_bias_anchor(
      bool enable, const std::optional<Covariance3>& R_vb = std::nullopt);

  /**
   * Reset transport J(delta_xi, delta_x); P <- J P J^T. Exposed for testing.
   *
   * Analytic derivative, at eps = delta_xi, of the error re-centring map
   * eps -> Local(xi_ref, phi(Exp(-delta_x), Retract(xi_ref, eps))) that the
   * group correction applies to the error. J -> I as the correction -> 0.
   */
  Eigen::Matrix<double, 18, 18> resetMatrix(
      const Eigen::Matrix<double, 18, 1>& delta_xi,
      const Eigen::Matrix<double, 18, 1>& delta_x) const;

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
  Covariance18 inputNoiseCov(const Eigen::Vector3d& w_meas,
                             const Eigen::Vector3d& a_meas,
                             const Eigen::Vector3d& g_vec,
                             const ImuNoise& noise) const;

  // DVL body-velocity update via h(xi) = R^T v, using the equivariant C*.
  void update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl);

  // Position update via h'(xi) = R^T(pi - p), using the equivariant C*.
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

  /**
   * Pressure-sensor depth update from the world-frame z position.
   *
   * The scalar model h(xi) = e_3^T p is not equivariant: under the state action
   * it picks up a term e_3^T R p_A that no output action can reproduce. The
   * depth is therefore stacked onto the estimated horizontal position to form
   * the pseudo-position p_tilde = [p_x, p_y, z_depth]^T, which is fed through
   * the position output with the horizontal variance inflated so only the
   * vertical direction is corrected. Since the horizontal entries come from the
   * estimate rather than the true state, this update is a second-order
   * approximation.
   *
   * @param z_depth              Measured world-frame z position.
   * @param R_depth              Variance of z_depth, as a 1x1 matrix.
   * @param horizontal_variance  Variance assigned to the pseudo-measured x and
   *                             y axes.
   */
  void update_depth(double z_depth, const Covariance1& R_depth,
                    double horizontal_variance = kDefaultHorizontalVariance);

  // Virtual-bias anchor b_v = 0 pseudo-measurement.
  void update_virtual_bias(const Covariance3& R_vb);

  State errorState(const State& xi_true) const;
  TangentVector errorStateVector(const State& xi_true) const;

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
   *
   * Cstar must already be in error coordinates, the tangent space at the
   * reference state, and prediction, z and R must share one output frame with
   * it.
   */
  void updateWithReset(const Eigen::VectorXd& prediction,
                       const Eigen::MatrixXd& Cstar, const Eigen::VectorXd& z,
                       const Eigen::MatrixXd& R);

  bool anchor_virtual_bias_ = true;

  // Process noise and measurement noise for the virtual-bias state.
  Covariance3 virtual_bias_R_ = 1e-6 * Covariance3::Identity();
  Covariance3 virtual_bias_Q_ = 1e-6 * Covariance3::Identity();

  bool reset_step_ = true;

  // Fixed-origin caches for the current reference state (xi_ref_).
  Eigen::Matrix<double, 18, 18> innovation_lift_;
  Eigen::Matrix<double, 18, 18> orbit_jacobian0_;
  Eigen::Matrix<double, 18, 18> input_lift_;
};

}  // namespace tgeqf
}  // namespace gtsam
