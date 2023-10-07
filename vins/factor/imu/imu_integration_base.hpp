//
// Created by weihao on 23-10-7.
//

#ifndef IMU_INTEGRATION_BASE_HPP
#define IMU_INTEGRATION_BASE_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include <vector>
#include "common/parameter.hpp"
#include <glog/logging.h>

namespace factor {
/// @brief IMU integration class
/// 1. Compute IMU integration, dp, dq, dv
/// 2. Compute jacobian for every state variable
/// 3. Compute convariance for system
class IntegrationBase
{
public:
  IntegrationBase() = delete;
  IntegrationBase(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0, const Eigen::Vector3d &linearized_ba,
                  const Eigen::Vector3d &linearized_bg);
  void ReadNoiseParameter();

public:
  Eigen::Vector3d acc_0_, gyr_0_;
  Eigen::Vector3d acc_1_, gyr_1_;

  Eigen::Vector3d linearized_acc_, linearized_gyr_;
  Eigen::Vector3d linearized_ba_, linearized_bg_;

  Eigen::Vector3d delta_p_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond delta_q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d delta_v_ = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 15, 15> jacobian_ = Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> covariance_ = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 18, 18> noise_ = Eigen::Matrix<double, 18, 18>::Zero();
  Eigen::Matrix<double, 15, 15> step_jacobian_;
  Eigen::Matrix<double, 15, 18> step_V_;

  std::vector<double> dt_buf_;
  std::vector<Eigen::Vector3d> acc_buf_;
  std::vector<Eigen::Vector3d> gyr_buf_;

  double sum_dt_ = 0.0;
  double dt_ = 0.0;

public:
  double ACC_N; /// accelerometer measurement noise standard deviation.
  double GYR_N; /// gyroscope measurement noise standard deviation.
  double ACC_W; /// accelerometer bias random work noise standard deviation.
  double GYR_W; /// gyroscope bias random work noise standard deviation.
};

} // namespace factor

#endif // IMU_INTEGRATION_BASE_HPP
