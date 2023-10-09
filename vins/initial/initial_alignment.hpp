//
// Created by weihao on 23-10-9.
//

#ifndef INITIAL_ALIGNMENT_HPP
#define INITIAL_ALIGNMENT_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include "factor/imu/imu_integration_base.hpp"
#include <vector>
#include <map>

namespace initial {
class ImageImuFrame
{
public:
  ImageImuFrame() = default;
  ImageImuFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &points, double t): t_{t}
  {
    points_ = points;
  };
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points_;
  double t_ = 0.;
  Eigen::Matrix3d R_;
  Eigen::Vector3d T_;
  factor::IntegrationBase *pre_integration_ = nullptr;
  bool is_key_frame_ = false;
  static void SolveGyroscopeBias(std::map<double, ImageImuFrame> &all_image_frame, Eigen::Vector3d *Bgs,
                                 int WINDOW_SIZE);
};
} // namespace initial

#endif // INITIAL_ALIGNMENT_HPP
