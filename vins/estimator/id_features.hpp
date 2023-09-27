//
// Created by weihao on 23-9-27.
//

#ifndef ID_FEATURES_HPP
#define ID_FEATURES_HPP

#include "estimator/frame_feature.hpp"
#include <vector>

namespace estimator {

class IDWithObservedFeatures
{
public:
  const int feature_id_;
  int start_frame_;
  std::vector<ObservedFrameFeature> feature_per_frame_;
  int used_num_;
  double estimated_depth_;
  int solve_flag_; /// 0 haven't solve yet; 1 solve success; 2 solve fail;

  IDWithObservedFeatures(int _feature_id, int _start_frame)
    : feature_id_(_feature_id), start_frame_(_start_frame), used_num_(0), estimated_depth_(-1.0), solve_flag_(0)
  {
  }

  int EndFrame() { return start_frame_ + static_cast<int>(feature_per_frame_.size()) - 1; }
};
} // namespace estimator

#endif //ID_FEATURES_HPP
