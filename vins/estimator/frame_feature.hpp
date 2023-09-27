//
// Created by weihao on 23-9-27.
//

#ifndef FRAME_FEATURE_HPP
#define FRAME_FEATURE_HPP

#include "Eigen/Core"
#include "Eigen/Dense"

namespace estimator {
class ObservedFrameFeature
{
public:
  ObservedFrameFeature(const Eigen::Matrix<double, 7, 1> &point, double td)
  {
    point_.x() = point(0);
    point_.y() = point(1);
    point_.z() = point(2);
    uv_.x() = point(3);
    uv_.y() = point(4);
    velocity_.x() = point(5);
    velocity_.y() = point(6);
    cur_td_ = td;
  }
  void RightObservation(const Eigen::Matrix<double, 7, 1> &point)
  {
    point_right_.x() = point(0);
    point_right_.y() = point(1);
    point_right_.z() = point(2);
    uv_right_.x() = point(3);
    uv_right_.y() = point(4);
    velocity_right_.x() = point(5);
    velocity_right_.y() = point(6);
  }
  double cur_td_;
  Eigen::Vector3d point_, point_right_;
  Eigen::Vector2d uv_, uv_right_;
  Eigen::Vector2d velocity_, velocity_right_;
  bool is_stereo_ = true;
};
} // namespace estimator

#endif //FRAME_FEATURE_HPP
