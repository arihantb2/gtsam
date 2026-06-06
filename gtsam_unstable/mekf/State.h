#pragma once
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Lie.h>
#include <Eigen/Dense>

// ============================================================================
//  MekfState : physical manifold state for the baseline Multiplicative EKF
// ----------------------------------------------------------------------------
//  xi = [R, v, p, b_gyro, b_accel]   on  M := SO(3) x R^12      dim = 15
//
//  Error convention (body / right multiplicative, additive on vectors):
//      R_true = R_hat * Exp(d_theta),  v = v_hat + d_v, ...
//  This matches both sibling filters (tg_eqf, tfg_inekf) so the error metrics
//  and NEES are directly comparable.
//
//  Tangent block order (fixed for ALL Jacobians / covariances in this module):
//      [ d_theta(3) ; d_v(3) ; d_p(3) ; d_b_gyro(3) ; d_b_accel(3) ]
//  identical to tfg_inekf so defaultQc()/CSV logging are reusable unchanged.
// ============================================================================

namespace mekf {

struct MekfState {
    gtsam::Rot3     R;        ///< SO(3): body orientation
    Eigen::Vector3d v;        ///< R^3:   velocity in global frame
    Eigen::Vector3d p;        ///< R^3:   position in global frame
    Eigen::Vector3d b_gyro;   ///< gyroscope     bias (body frame)
    Eigen::Vector3d b_accel;  ///< accelerometer bias (body frame)

    static constexpr int dimension = 15;

    /// Identity/origin state: R=I, all vectors zero.
    MekfState();

    /// Direct constructor.
    MekfState(const gtsam::Rot3& R, const Eigen::Vector3d& v,
              const Eigen::Vector3d& p, const Eigen::Vector3d& b_gyro,
              const Eigen::Vector3d& b_accel);

    static MekfState identity();
};

} // namespace mekf

// ---------------------------------------------------------------------------
// GTSAM traits: model the Manifold concept so MekfState can be the template
// parameter M of gtsam::ManifoldEKF<M>.
// ---------------------------------------------------------------------------
namespace gtsam {

template <>
struct traits<mekf::MekfState> {
    using ManifoldType       = mekf::MekfState;
    using TangentVector      = Eigen::Matrix<double, 15, 1>;
    using Jacobian           = Eigen::Matrix<double, 15, 15>;
    using structure_category = manifold_tag;

    static constexpr int dimension = 15;
    static int GetDimension(const mekf::MekfState&) { return dimension; }

    /// Retraction at xi: R*Exp(d_theta), additive on vectors.
    /// delta order = [d_theta(3); d_v(3); d_p(3); d_b_gyro(3); d_b_accel(3)].
    static mekf::MekfState Retract(const mekf::MekfState& xi,
                                   const TangentVector& delta);

    /// Local(xi_ref, xi): inverse of Retract (error in the tangent at xi_ref).
    static TangentVector Local(const mekf::MekfState& xi_ref,
                               const mekf::MekfState& xi);

    static mekf::MekfState Identity() { return mekf::MekfState::identity(); }

    static bool Equals(const mekf::MekfState& a, const mekf::MekfState& b,
                       double tol = 1e-9);
    static void Print(const mekf::MekfState& xi, const std::string& s = "");
};

template <>
struct traits<const mekf::MekfState> : traits<mekf::MekfState> {};

} // namespace gtsam
