#ifndef FACTOR_PROJECTION_ONE_FRAME_TWO_CAMERA_HPP
#define FACTOR_PROJECTION_ONE_FRAME_TWO_CAMERA_HPP

#include <ceres/ceres.h>
#include <Eigen/Core>
#include "common/algorithm.hpp"

namespace factor {
class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1, 1>
{
public:
  ProjectionOneFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                 const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                 const double _td_i, const double _td_j);

  virtual bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;
  void ComputeResidual(const std::vector<const double *> parameters) const;

  Eigen::Vector3d pts_i_, pts_j_;
  Eigen::Vector3d velocity_i_, velocity_j_;
  double td_i_, td_j_;
  static Eigen::Matrix2d sqrt_info_;
  static bool cout_residual_;
};
} // namespace factor

#endif // FACTOR_PROJECTION_ONE_FRAME_TWO_CAMERA_HPP