#pragma once
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Lie.h>
#include <Eigen/Dense>

// ============================================================================
//  Two-Frames Group  G_TF = SO(3) |x (R^6 (+) R^6)        dim = 15
// ----------------------------------------------------------------------------
//  Symmetry group of the biased INS problem used by the TFG-IEKF.
//
//  Reference: Fornasier et al., "Equivariant Symmetries for Inertial
//  Navigation Systems" (arXiv:2309.03765v3).
//    - Section 5.3            : definition of G_TF and its action
//    - Table 2               : TFG-IEKF row (linearised error dynamics)
//    - Appendix B.7          : full TFG-IEKF derivation (B.31)-(B.35)
//    - Barrau & Bonnabel [4] : original two-frames group
//
//  An element is X = (C, gamma) with
//    C     = (A, (a, b)) in SE_2(3)   navigation part (rotation + 2 vectors)
//    gamma = (g_w, g_a) in R^6        bias part, lives in the SO(3)-rotated
//                                     "second frame" {I} (Sec. 5.3)
//
//  Group product (Sec. 5.3, p.7):
//    X*Y      = ( C_X C_Y , gamma_X + A_X * gamma_Y )
//    X^{-1}   = ( C^{-1}  , -A^T * gamma )
//  where A * (x1, x2) := (A x1, A x2) rotates each R^3 block (Sec. 5.3).
//
//  Relation to the physical state xi = (T, b),  T in SE_2(3), b in R^6:
//    the filter state X acts on the fixed origin xi0 = (I_5, 0) via the action
//    phi (see Symmetry.h), giving  T_hat = C,  b_hat = A^{-1} * (-gamma)   (B.31)
//  We therefore store the navigation part directly as (R, v, p) and keep the
//  bias generator gamma; physical biases are recovered with bias_omega() /
//  bias_accel(), and a state is packed into a group element with FromState().
// ============================================================================

namespace tfg {

using gtsam::Rot3;

/// Lie algebra element of g_TF = so(3) (+) R^6 (+) R^6, as a 15-vector.
/// Block order (fixed for ALL Jacobians/covariances in this module):
///   [ d_theta(3) ; d_v(3) ; d_p(3) ; d_g_omega(3) ; d_g_accel(3) ]
/// The leading 9 blocks match gtsam SE_2(3) Logmap ordering (rot, vel, pos).
using Tangent = Eigen::Matrix<double, 15, 1>;

class TwoFrameGroup {
public:
    static constexpr int dim = 15;
    using TangentVector = Tangent;
    using Jacobian      = Eigen::Matrix<double, dim, dim>;

    // --- Data: navigation part C = (R, v, p) and bias generator gamma -------
    Rot3            R;          ///< A   in SO(3)
    Eigen::Vector3d v;          ///< a   in R^3   (SE_2(3) velocity block)
    Eigen::Vector3d p;          ///< b   in R^3   (SE_2(3) position block)
    Eigen::Vector3d gamma_omega;///< g_w in R^3   (gyro  bias generator)
    Eigen::Vector3d gamma_accel;///< g_a in R^3   (accel bias generator)

    /// Identity element  (I, 0, 0, 0, 0).
    TwoFrameGroup();

    /// Direct constructor from group coordinates.
    TwoFrameGroup(const Rot3& R, const Eigen::Vector3d& v,
                  const Eigen::Vector3d& p, const Eigen::Vector3d& gamma_omega,
                  const Eigen::Vector3d& gamma_accel);

    // --- Conversion to/from the physical state (B.31) -----------------------
    /// Build a group element from a physical state (T, b): gamma = -R * b.
    static TwoFrameGroup FromState(const Rot3& R, const Eigen::Vector3d& v,
                                   const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& b_omega,
                                   const Eigen::Vector3d& b_accel);

    /// Recover physical biases  b = A^{-1} * (-gamma)  (B.31).
    Eigen::Vector3d bias_omega() const;
    Eigen::Vector3d bias_accel() const;

    // --- Group operations (Sec. 5.3) ----------------------------------------
    static TwoFrameGroup Identity();
    TwoFrameGroup operator*(const TwoFrameGroup& Y) const;   ///< X*Y
    TwoFrameGroup inverse() const;                            ///< X^{-1}

    // --- Lie group maps -----------------------------------------------------
    /// Exp: g_TF -> G_TF. SE_2(3) part is exact; bias blocks are the trivial
    /// (R^6) exponential rotated into the second frame.
    static TwoFrameGroup Expmap(const TangentVector& xi);
    /// Log: G_TF -> g_TF (inverse of Expmap).
    TangentVector Logmap() const;

    /// Adjoint Ad_X : g_TF -> g_TF, 15x15. Required by InvariantEKF for the
    /// covariance propagation  P <- Ad_{U^{-1}} P Ad_{U^{-1}}^T + Q.
    Jacobian AdjointMap() const;

    /// 5x5 matrix realisation of the SE_2(3) navigation part C.
    Eigen::Matrix<double, 5, 5> C_matrix() const;
};

} // namespace tfg

// ---------------------------------------------------------------------------
// GTSAM traits: make TwoFrameGroup model the LieGroup concept so it can be the
// template parameter G of gtsam::InvariantEKF<G>.
// ---------------------------------------------------------------------------
namespace gtsam {

template <>
struct traits<tfg::TwoFrameGroup> {
    using ManifoldType       = tfg::TwoFrameGroup;
    using TangentVector      = tfg::Tangent;
    using Jacobian           = Eigen::Matrix<double, 15, 15>;
    using ChartJacobian      = OptionalJacobian<15, 15>;
    using structure_category = lie_group_tag;
    using group_flavor       = multiplicative_group_tag;

    static constexpr int dimension = 15;
    static int GetDimension(const tfg::TwoFrameGroup&) { return dimension; }

    // Testable
    static bool Equals(const tfg::TwoFrameGroup& X, const tfg::TwoFrameGroup& Y,
                       double tol = 1e-9);
    static void Print(const tfg::TwoFrameGroup& X, const std::string& s = "");

    // Group
    static tfg::TwoFrameGroup Identity();
    static tfg::TwoFrameGroup Compose(const tfg::TwoFrameGroup& X,
                                      const tfg::TwoFrameGroup& Y,
                                      ChartJacobian Hx = {}, ChartJacobian Hy = {});
    static tfg::TwoFrameGroup Between(const tfg::TwoFrameGroup& X,
                                      const tfg::TwoFrameGroup& Y,
                                      ChartJacobian Hx = {}, ChartJacobian Hy = {});
    static tfg::TwoFrameGroup Inverse(const tfg::TwoFrameGroup& X,
                                      ChartJacobian H = {});

    // Lie group
    static tfg::TwoFrameGroup Expmap(const TangentVector& xi, ChartJacobian H = {});
    static TangentVector      Logmap(const tfg::TwoFrameGroup& X, ChartJacobian H = {});
    static Jacobian           AdjointMap(const tfg::TwoFrameGroup& X);

    // Manifold
    static tfg::TwoFrameGroup Retract(const tfg::TwoFrameGroup& X,
                                      const TangentVector& xi);
    static TangentVector      Local(const tfg::TwoFrameGroup& X,
                                    const tfg::TwoFrameGroup& Y);
};

template <>
struct traits<const tfg::TwoFrameGroup> : traits<tfg::TwoFrameGroup> {};

} // namespace gtsam
