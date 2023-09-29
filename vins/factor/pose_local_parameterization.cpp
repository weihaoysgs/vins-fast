#include "factor/pose_local_parameterization.hpp"

namespace factor {

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> p(x);
  Eigen::Map<const Eigen::Quaterniond> q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);
  Eigen::Quaterniond dq = common::Algorithm::DeltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> update_p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> update_q(x_plus_delta + 3);

  update_p = p + dp;
  update_q = (q * dq).normalized();
  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
  j.topRows(6).setIdentity();
  j.bottomRows(1).setZero();
  return true;
}

} // namespace factor
