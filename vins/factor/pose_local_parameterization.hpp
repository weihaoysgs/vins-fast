#include <ceres/ceres.h>
#include "Eigen/Core"
#include "common/algorithm.hpp"

namespace factor {
class PoseLocalParameterization : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 7; };
  virtual int LocalSize() const { return 6; };
};
} // namespace factor