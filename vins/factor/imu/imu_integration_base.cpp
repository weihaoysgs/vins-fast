//
// Created by weihao on 23-10-7.
//

#include "factor/imu/imu_integration_base.hpp"

namespace factor {
IntegrationBase::IntegrationBase(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,
                                 const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)

  : acc_0_{acc_0}
  , gyr_0_{gyr_0}
  , linearized_acc_{acc_0}
  , linearized_gyr_{gyr_0}
  , linearized_ba_{linearized_ba}
  , linearized_bg_{linearized_bg}
{
  ReadNoiseParameter();
  noise_.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
}

void IntegrationBase::ReadNoiseParameter()
{
  ACC_N = common::Setting::getSingleton()->Get<double>("acc_n");
  ACC_W = common::Setting::getSingleton()->Get<double>("acc_w");
  GYR_N = common::Setting::getSingleton()->Get<double>("gyr_n");
  GYR_W = common::Setting::getSingleton()->Get<double>("gyr_w");
  // LOG(INFO) << "ACC_N: " << ACC_N << " ACC_W: " << ACC_W << " GYR_N: " << GYR_N << " GYR_W: " << GYR_W;
}

} // namespace factor
