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
 * at the fixed reference state xi_ref. Measurements therefore come in two
 * flavours here, and the distinction is visible at the call site:
 *
 * - An ordinary measurement function differentiated at the current estimate
 *   (DVL, virtual bias) yields a state-chart Jacobian and goes through
 *   updateFromStateJacobian(), which transports it.
 * - An equivariant output matrix built at the reference state (position, C* of
 *   Equ. (B.19)) is already in error coordinates and goes straight to
 *   updateWithReset().
 *
 * Getting this backwards is silent: both matrices have the same shape and
 * coincide when the group estimate is the identity.
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

  /// Initial stddev of the virtual bias b_v. The virtual bias has no physical
  /// counterpart to perturb: it starts at zero, held there by the b_v = 0
  /// anchor, so its initial uncertainty matches that anchor rather than any
  /// user-set bias sigma.
  static constexpr double kVirtualBiasInitialSigma = 1e-6;

  /// Full initial covariance built from the physical-state one, ordered
  /// [R, v, p, b_w, b_a]. The physical block is used as given; the virtual-bias
  /// block is kVirtualBiasInitialSigma^2 * I and uncorrelated with the rest.
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
   * J -> I as the correction -> 0.
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

  // DVL body-velodity update: h(xi) = R^T v
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
   * approximation. See Sec. V-B of docs/EqF_design_for_DVL_Depth_aided_INS.pdf.
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
   * reference state. Measurements holding a Jacobian at the current estimate
   * must use updateFromStateJacobian() instead.
   */
  void updateWithReset(const Eigen::VectorXd& prediction,
                       const Eigen::MatrixXd& Cstar, const Eigen::VectorXd& z,
                       const Eigen::MatrixXd& R);

  /**
   * Measurement update from a Jacobian taken in the chart at the current state
   * estimate, transported to error coordinates on the way in. This is the only
   * place the transport happens.
   */
  template <int Dim>
  void updateFromStateJacobian(
      const Eigen::Matrix<double, Dim, 1>& prediction,
      const Eigen::Matrix<double, Dim, 18>& H_at_estimate,
      const Eigen::Matrix<double, Dim, 1>& z,
      const Eigen::Matrix<double, Dim, Dim>& R) {
    updateWithReset(prediction, outputMatrix(H_at_estimate), z, R);
  }

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
