#pragma once
#include <gtsam/navigation/EquivariantFilter.h>
#include <gtsam_unstable/tg_eqf/State.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <Eigen/Dense>
#include <optional>

namespace tgeqf {

// Continuous-time IMU noise PSDs; mapped through lift differential B
struct ImuNoise {
  double gyro = 0.0;
  double accel = 0.0;
  double gyro_rw = 0.0;
  double accel_rw = 0.0;
};

// TG-EqF biased INS filter on G = SE_2(3) ⋉ se_2(3)
class TGEqF : public gtsam::EquivariantFilter<State, TGSymmetry> {
 public:
  using Base = gtsam::EquivariantFilter<State, TGSymmetry>;
  using Covariance18 = Eigen::Matrix<double, 18, 18>;
  using Covariance3 = Eigen::Matrix<double, 3, 3>;

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

  // Position update via h'(xi) = R^T(pi - p).
  void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos,
                       bool use_Cstar = true);

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
  // Dphi_g = d phi(g, xi)/dxi: H_origin = H_est * Dphi_g.
  // At g = identity this is I_18.
  Eigen::Matrix<double, 18, 18> originChartTransport() const;

  // Compute the fixed-origin caches (orbit_jacobian0_, innovation_lift_,
  // input_lift_) from the current referenceState(). Called on construction.
  void computeOriginCaches();

  void updateWithReset(const Eigen::VectorXd& prediction,
                       const Eigen::MatrixXd& H, const Eigen::VectorXd& z,
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
