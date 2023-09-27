//
// Created by weihao on 23-9-27.
//

#ifndef FEATURE_MANAGER_HPP
#define FEATURE_MANAGER_HPP

#include "estimator/id_features.hpp"
#include <list>

namespace estimator
{
class FeatureManager
{
public:
  FeatureManager();
  void SetRic(Eigen::Matrix3d ric[]);

private:
  std::list<IDWithObservedFeatures> features_;
  std::vector<Eigen::Matrix3d> ric_;
};

}

#endif // FEATURE_MANAGER_HPP
