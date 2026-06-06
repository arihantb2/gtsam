#pragma once
#include "Group.h"
#include <Eigen/Dense>

// ============================================================================
//  Direct global position output for the TFG-IEKF
// ----------------------------------------------------------------------------
//  Reference: Fornasier et al. (arXiv:2309.03765v3), Section 7
//    - Eq. (33)            : raw measurement  h(xi) = p in R^3
//    - Lemma 15, Eq.(34-35): equivariant body-frame reformulation + output action
//    - Appendix B.7, (B.35): equivariant output matrix C* for G_TF
//
//  A GNSS-style sensor gives a global position fix  pi in R^3. The raw model
//  h(xi) = p is NOT output-equivariant for G_TF (Sec. 7). Following Lemma 15 we
//  use the body-referenced residual model, which IS equivariant and yields a
//  third-order (cubic) linearisation error:
//
//      h(xi) = R^T (pi - p)               in R^3            (Eq. 34)
//      rho_X(y) = A^T (y - b)             output action     (Eq. 35)
//
//  Here (R, p) are the orientation/position of the state and (A, b) come from
//  the SE_2(3) part of the group element X = (C=(A,(a,b)), gamma).
//  Noise-free value of h is zero when pi == p (filter consistent).
// ============================================================================

namespace tfg {

struct PositionOutput {
    /// Output dimension (full 3-DoF position).
    static constexpr int dim = 3;

    /// Output Jacobian variant.
    enum class Variant {
      C0,     ///< first-order Jacobian of the equivariant residual (Eq. B.35 w/o averaging)
      Cstar,  ///< midpoint-symmetrised, cubic linearisation error (Eq. B.35)
    };

    /**
     * Body-frame residual measurement  h(xi) = R^T (pi - p)   (Eq. 34).
     * @param xi  current state (group element; R, p read from its nav part)
     * @param pi  raw global position fix
     */
    static Eigen::Vector3d predict(const TwoFrameGroup& xi,
                                   const Eigen::Vector3d& pi);

    /**
     * Equivariant output action  rho_X(y) = R_X^T (y - p_X)   (Eq. 35).
     * Only used to check/exploit output equivariance:
     *   h(phi(X,xi)) = h(xi * X) = rho_X(h(xi)).
     */
    static Eigen::Vector3d output_action(const TwoFrameGroup& X,
                                         const Eigen::Vector3d& y);

    /**
     * Output Jacobian C in R^{3x15} for the right-retract chart
     * xi(eps) = xi_hat * Expmap(eps). Block order matches Group.h::Tangent.
     *
     * By output equivariance h(xi(eps)) = rho_{Exp(eps)}(y_hat) exactly, with
     * y_hat = h(xi_hat) = R_hat^T (pi - p_hat). Differentiating at eps = 0:
     *
     *   C0    = [ y_hat^             0_3  -I_3  0_{3x6} ]   (true 1st-order)
     *   Cstar = [ 1/2 (y_hat+p_hat)^ 0_3  -I_3  0_{3x6} ]  (Eq. B.35)
     *
     * Cstar averages the skew of the predicted output y_hat with that of the
     * back-transported measurement rho_{X_hat^-1}(0) = p_hat, centring the
     * attitude coupling between estimate and measurement. This yields cubic
     * (O(||e||^3)) rather than quadratic linearisation error and improves the
     * update when the innovation is large (Fornasier Sec. 7.1, Eq. B.35).
     *
     * @param xi_hat  current state estimate
     * @param pi      raw global position fix
     * @param variant Cstar (default) or C0
     */
    static Eigen::Matrix<double, 3, 15> jacobian(
        const TwoFrameGroup& xi_hat, const Eigen::Vector3d& pi,
        Variant variant = Variant::Cstar);

    /**
     * Innovation for the EKF update. The body-frame residual model has
     * noise-free target z = 0, so the innovation is  z - h(xi_hat) = -y_hat.
     * (Sign convention to be matched against InEKF::update_position.)
     */
    static Eigen::Vector3d innovation(const TwoFrameGroup& xi_hat,
                                      const Eigen::Vector3d& pi);
};

} // namespace tfg
