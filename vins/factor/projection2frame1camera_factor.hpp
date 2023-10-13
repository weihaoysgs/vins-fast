#include <ceres/ceres.h>
#include "Eigen/Core"
#include "common/algorithm.hpp"

namespace factor {
class ProjectionTwoFrameOneCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
  ProjectionTwoFrameOneCamFactor(const Eigen::Vector3d &pts_i,const Eigen::Vector3d &pts_j,
                                 const Eigen::Vector2d &vel_i, const Eigen::Vector2d &vel_j,
                                 const double td_i, const double td_j);

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobi) const ;
  void CheckPoseJacobi(const Eigen::Matrix<double, 2, 7>& analysis_jacobi,
                       const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi,
                       const Eigen::Matrix3d &Qj, const Eigen::Vector3d &Pj,
                       const Eigen::Vector3d &pts_i_td, const Eigen::Vector3d &pts_j_td,
                       double inv_dep_i, const Eigen::Quaterniond &qic,
                       const Eigen::Vector3d &tic, const Eigen::Vector2d &residual) const;
  void ComputeResidual(const std::vector<const double *> parameters) const;

  /// 观测在优化过程中是不会变的，包括2d点和速度
  Eigen::Vector3d pts_i_, pts_j_;
  Eigen::Vector3d velocity_i_, velocity_j_;
  double td_i_, td_j_;
  static Eigen::Matrix2d sqrt_info_;
  static bool cout_residual_;
};
} // namespace factor