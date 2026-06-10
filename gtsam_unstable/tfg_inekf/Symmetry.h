#pragma once
#include "Group.h"
#include <Eigen/Dense>

// ============================================================================
//  Two-Frames symmetry of the biased INS:  action phi + lift Lambda
// ----------------------------------------------------------------------------
//  Reference: Fornasier et al. (arXiv:2309.03765v3)
//    - Section 4, Eq. (3)   : continuous-time biased INS dynamics
//    - Section 5.3          : group action phi (Lemma 5, Eq. 16)
//    - Theorem 6, Eq.(17-18): equivariant lift Lambda for G_TF
//    - Appendix B.7         : resulting TFG-IEKF error/state matrices
//
//  IMU input (Eq. 5):  u = (omega, accel, tau_omega, tau_accel)
//    omega, accel        : gyro / accelerometer readings (body frame {I})
//    tau_omega, tau_accel: bias random-walk drivers (zero for constant bias)
//
//  The lift turns the system dynamics (3) into a curve on the group so that the
//  left-invariant EKF can propagate the state by group composition,
//    X_{k+1} = X_k * U,   U = Expmap(Lambda(X_k, u) * dt).
//  Because Lambda depends on the state (the biases), the increment U is built
//  here and passed to gtsam::InvariantEKF::predict(U, Q) (not the
//  state-independent predict(u, dt, Q) overload).
// ============================================================================

namespace tfg {

/// Lifted IMU input, fed to InvariantEKF::predict.
/// Mirrors u in Eq. (5); biases are part of the state, not this struct.
struct ImuInput {
    Eigen::Vector3d omega;       ///< gyroscope reading      (body frame)
    Eigen::Vector3d accel;       ///< accelerometer reading  (body frame)
    Eigen::Vector3d tau_omega = Eigen::Vector3d::Zero(); ///< gyro  bias driver
    Eigen::Vector3d tau_accel = Eigen::Vector3d::Zero(); ///< accel bias driver
};

/// Continuous-time IMU noise PSDs (variances), used to build the lifted process
/// noise via the input matrix B (see inputNoiseCov). Each entry is the per-axis
/// PSD; isotropic per channel.
struct ImuNoise {
    double gyro = 0.0;      ///< gyroscope white-noise PSD      (rad^2/s)
    double accel = 0.0;     ///< accelerometer white-noise PSD  (m^2/s^3)
    double gyro_rw = 0.0;   ///< gyro  bias random-walk PSD      (rad^2/s^3)
    double accel_rw = 0.0;  ///< accel bias random-walk PSD      (m^2/s^5)
};

/// Gravity magnitude (m/s^2). Single source of truth shared by gravity() and
/// the example IMU simulators, so the filter model and the data generator
/// cannot drift apart.
inline constexpr double kGravity = 9.81;

/// Gravity in the global frame {G} (Eq. 3b), default -z.
Eigen::Vector3d gravity();

// ----------------------------------------------------------------------------
//  Right group action  phi : G_TF x M -> M           (Sec. 5.3, Eq. 16)
//    phi(X, xi) = ( T C , A^T * (b - gamma) )
//  Transitive, so the origin xi0 = (I_5, 0) generates the whole state space.
//  Used to (a) recover the state estimate from X, (b) document equivariance.
// ----------------------------------------------------------------------------
TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi);

// ----------------------------------------------------------------------------
//  Equivariant lift  Lambda : M x L -> g_TF          (Theorem 6, Eq. 17-18)
//
//    Lambda_1(xi,u) = (W - B + N) + T^{-1}(G - N) T          in se_2(3)
//    Lambda_2(xi,u) = ( b_w^ (omega - b_w) - tau_w ,
//                       b_a^ (omega - b_w) - tau_a )         in R^6
//
//  with W, B, N, G the se_2(3) matrices built from (omega, accel) and the
//  biases (Sec. 5.2, Eq. 12). In vee coordinates Lambda_1 reduces to the closed
//  form of Eq. 3 (theta = omega - b_w, v_dot = (a - b_a) + R^T g, p_dot = R^T v),
//  which is what the implementation evaluates. Returns the 15-vector in g_TF
//  (block order of Group.h::Tangent), the per-step generator for InvariantEKF.
// ----------------------------------------------------------------------------
Tangent lift(const TwoFrameGroup& X, const ImuInput& u);

/// Jacobian Df = d lift(X * Expmap(eps), u) / d eps, evaluated at eps = 0, in
/// the right-retract chart (tangent block order of Group.h::Tangent). Captures
/// the state-dependence of the lift on the biases; this is the coupling that
/// turns the plain Ad-only covariance propagation into the TFG-IEKF error
/// dynamics (Fornasier Table 2 / Eq. B.34). Returns a 15x15 matrix.
Eigen::Matrix<double, 15, 15> liftJacobian(const TwoFrameGroup& X,
                                           const ImuInput& u);

/// Discrete process-noise covariance Qd = B(X) diag(sigma^2) B(X)^T * dt, where
/// B maps the IMU input/bias-driver noise (omega, accel, tau_w, tau_a) into the
/// lifted tangent. Non-trivial because gyro noise enters the bias-generator
/// rows through the lift (Eq. 18): B[theta,n_w]=I, B[v,n_a]=I, B[g_w,n_w]=b_w^,
/// B[g_w,tau_w]=-I, B[g_a,n_w]=b_a^, B[g_a,tau_a]=-I. Returns a 15x15 matrix.
Eigen::Matrix<double, 15, 15> inputNoiseCov(const TwoFrameGroup& X,
                                            const ImuNoise& noise, double dt);

/// Convenience: group increment  U = Expmap(lift(X,u) * dt)  for
/// InvariantEKF::predict(U, Q). Keeps the EKF call site symmetry-agnostic.
TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt);

} // namespace tfg
