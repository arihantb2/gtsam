#pragma once
#include <gtsam/base/TangentLieGroup.h>
#include <gtsam/geometry/ExtendedPose3.h>

namespace gtsam {
namespace tgeqf {

/// Group element of G = SE_2(3) ⋉ se_2(3), the tangent group of SE_2(3) under
/// its adjoint action: pairs (A, a) with A in SE_2(3), a in se_2(3), and
/// product (A1,a1)(A2,a2) = (A1 A2, a1 + Ad_{A1} a2).
using TGElement = gtsam::TangentLieGroup<gtsam::ExtendedPose3<2>>;

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

template <>
struct traits<const tgeqf::TGElement> : traits<tgeqf::TGElement> {};

}  // namespace gtsam
