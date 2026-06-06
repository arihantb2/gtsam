#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Extended input vector u in L = R^21
 * u = [w, tau, g_vec]
 * w     = [omega, v_tilde, accel] in R^9   (IMU + virtual inputs)
 * tau   = [tau_omega, tau_v, tau_a]  in R^9   (bias rate inputs)
 * g_vec = gravity vector             in R^3
 *
 * Reference: proposal Eq. (9), Section 6.3
 */
struct TGInput {
    Eigen::Vector3d omega;     // measured angular velocity
    Eigen::Vector3d v_tilde;   // virtual velocity input (set to b_v estimate)
    Eigen::Vector3d accel;     // measured linear acceleration
    Eigen::Vector3d tau_omega; // bias rate for b_omega (usually zero)
    Eigen::Vector3d tau_v;     // bias rate for b_v     (usually zero)
    Eigen::Vector3d tau_a;     // bias rate for b_a     (usually zero)
    Eigen::Vector3d g_vec;     // gravity vector in global frame

    /// Pack to R^21
    Eigen::Matrix<double, 21, 1> vector() const;

    /// Unpack from R^21
    static TGInput from_vector(const Eigen::Matrix<double, 21, 1>& v);
};

/**
 * Lift functor Lambda(xi, u) : M x L -> g
 *
 * Theorem 9 (Eq. 21-22) form:
 *   Lambda_1(xi, u) = (W - B + N) + T^{-1}(G - N)T            in se_2(3)
 *   Lambda_2(xi, u) = ad_b[Lambda_1(xi, u)] - tau^            in se_2(3)
 *
 * Equivalent form used in this implementation:
 *   Lambda_1(xi, u) = (W - B) + T^{-1} G T + T^{-1} f1(T)     in se_2(3)
 *   Lambda_2(xi, u) = ad_b[Lambda_1(xi, u)] - tau^            in se_2(3)
 *
 * The two Lambda_1 forms are identical via the identity
 *   N - T^{-1} N T = T^{-1} f1(T),   equivalently   T N - N T = f1(T)
 * where N is the 5x5 coupling matrix (Eq. 12) encoding dp = v, and f1(T)
 * is the velocity-drift matrix (Eq. 5): the 5x5 matrix carrying the global
 * velocity v in the position column. Also Ad_{T^{-1}}[G] = T^{-1} G T for the
 * SE_2(3) adjoint (matrix conjugation in the Lie algebra).
 *
 * Note: gtsam's ExtendedPose3<2> uses column order T = (R, p, v) (position in
 * column 4, velocity in column 5), swapped vs the paper's T = (R, v, p); the
 * implementation is internally consistent with its own convention.
 *
 * Satisfies equivariance: Lambda(phi(X,xi), psi(X,u)) = Ad_{X^{-1}} Lambda(xi,u)
 *
 * Reference: Fornasier et al. [1] Theorem 9 (Eq. 21-22), proposal Section 6.5 (TODO)
 *
 * GTSAM concept requirement:
 *   Lift(u)(xi, D_lift) -> TangentG (R^18, the Lie algebra g)
 *   where D_lift = d(Lambda)/d(xi) in R^{18 x 18} (dim g x dim M)
 */
struct Lift {
    TGInput u;

    explicit Lift(const TGInput& u);

    /// Lambda(xi, u) with optional Jacobian d(Lambda)/d(xi) in R^{18 x 18}
    Eigen::Matrix<double, 18, 1> operator()(
        const TGState& xi,
        Eigen::Matrix<double, 18, 18>* D_lift = nullptr) const;

private:
    /// W = wedge(w): input twist (Eq. 8)
    Eigen::Matrix<double, 5, 5> W_matrix(const TGInput& u) const;
    /// B = wedge(b): bias twist (Eq. 8)
    Eigen::Matrix<double, 5, 5> B_matrix(const TGState& xi) const;
    /// G = wedge(0,0,g): gravity in the acceleration slot (Eq. 8)
    Eigen::Matrix<double, 5, 5> G_matrix(const TGInput& u) const;
    /// f1(T) = velocity drift (Eq. 5): global velocity v in the position column
    Eigen::Matrix<double, 5, 5> f1_matrix(const TGState& xi) const;
};

/**
 * Input orbit functor psi_u : G -> L
 *
 * psi(X, u) maps the current input to the origin via the input symmetry action.
 * Used in EquivariantFilter::computeErrorDynamicsMatrix as:
 *   u_origin = psi_u(g_.inverse())
 *
 * Reference: van Goor et al. [6] Eq. (13), equivariance of the lift
 * Fornasier 2023 Lemma 8 Eq. (20) for G_TG = SE_2(3) ⋉ se_2(3):
 *   psi(X, u)_w   = Ad_{C^{-1}}(w - a^vee) + Omega(C^{-1})   in R^9
 *   psi(X, u)_tau = Ad_{C^{-1}}(tau)                          in R^9
 *   psi(X, u)_g   = g_vec                                      (unchanged)
 *
 * where C = SE_2(3) part of X, a = se_2(3) part of X, w and tau from u,
 * and Omega(C^{-1}) = (0, 0, a) places C^{-1}'s position in the accel slot.
 */
struct InputOrbit {
    TGInput u;

    explicit InputOrbit(const TGInput& u);

    /// psi(X, u): maps group element X to transformed input in L
    TGInput operator()(const TGGroupElement& X) const;
};

} // namespace tgeqf