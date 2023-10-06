//
// Created by weihao on 23-10-6.
//
#include "factor/marginalization/marg_factor.hpp"

namespace factor {
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *marginalization_info)
  : marginalization_info_(marginalization_info)
{
  for (auto it : marginalization_info_->keep_block_size_)
  {
    mutable_parameter_block_sizes()->push_back(it);
  }
  set_num_residuals(marginalization_info_->n_);
}

bool MarginalizationFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobi) const
{
  int n = marginalization_info_->n_;
  int m = marginalization_info_->m_;
  Eigen::VectorXd dx(n);
  for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++)
  {
    int size = marginalization_info_->keep_block_size_[i];
    int idx = marginalization_info_->keep_block_idx_[i] - m;
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
    Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info_->keep_block_data_[i], size);
    if (size != 7)
      dx.segment(idx, size) = x - x0;
    else
    {
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      dx.segment<3>(idx + 3) = 2.0 * common::Algorithm::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
      {
        dx.segment<3>(idx + 3) = 2.0 * -common::Algorithm::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      }
    }
  }
  Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info_->linearized_residuals_ + marginalization_info_->linearized_jacobi_ * dx;
  if (jacobi)
  {
    for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++)
    {
      if (jacobi[i])
      {
        int size = marginalization_info_->keep_block_size_[i], local_size = marginalization_info_->LocalSize(size);
        int idx = marginalization_info_->keep_block_idx_[i] - m;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobi[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) = marginalization_info_->linearized_jacobi_.middleCols(idx, local_size);
      }
    }
  }
  return true;
}
} // namespace factor
