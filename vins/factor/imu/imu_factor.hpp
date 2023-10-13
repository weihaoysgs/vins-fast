//
// Created by weihao on 23-10-7.
//

#ifndef IMU_FACTOR_HPP
#define IMU_FACTOR_HPP

#include "factor/imu/imu_integration_base.hpp"
#include <ceres/ceres.h>

namespace factor {
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
  IMUFactor() = delete;
  IMUFactor(IntegrationBase *pre_integration): pre_integration_(pre_integration){};
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void ComputeResidual(const std::vector<const double *> parameters) const;
  IntegrationBase *pre_integration_;
  static bool cout_imu_residual_;
};
} // namespace factor

#endif //IMU_FACTOR_HPP
