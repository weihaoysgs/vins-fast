#include "estimator/feature_manager.hpp"

namespace estimator {

FeatureManager::FeatureManager()
{
  for (int i = 0; i < 2; i++)
    ric_[i].setIdentity();
}

void FeatureManager::SetRic(Eigen::Matrix3d *ric)
{
  {
    for (int i = 0; i < 2; i++)
    {
      ric_[i] = ric[i];
    }
  }
}

} // namespace estimator
