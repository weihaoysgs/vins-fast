//
// Created by weihao on 23-10-6.
//

#ifndef MARG_RESIDUAL_BLOCK_INFO_HPP
#define MARG_RESIDUAL_BLOCK_INFO_HPP

#include <ceres/ceres.h>

#include <utility>
#include "Eigen/Core"

namespace factor {

/// @brief Simulate ceres residual block
struct MargResidualBlockInfo
{
  MargResidualBlockInfo(ceres::CostFunction *cost_function, ceres::LossFunction *loss_function,
                        std::vector<double *> parameter_blocks, std::vector<int> drop_set)
    : cost_function_(cost_function)
    , loss_function_(loss_function)
    , parameter_blocks_(std::move(parameter_blocks))
    , drop_set_(std::move(drop_set)){};
  void Evaluate();

  ceres::CostFunction *cost_function_;
  ceres::LossFunction *loss_function_;
  std::vector<double *> parameter_blocks_;
  std::vector<int> drop_set_;
  double **raw_jacobi_;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobi_;
  Eigen::VectorXd residuals_;

  int LocalSize(int size) { return size == 7 ? 6 : size; }
};

} // namespace factor

#endif // MARG_RESIDUAL_BLOCK_INFO_HPP
