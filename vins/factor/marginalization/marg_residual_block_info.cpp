//
// Created by weihao on 23-10-6.
//
#include "factor/marginalization/marg_residual_block_info.hpp"

namespace factor {
void MargResidualBlockInfo::Evaluate()
{
  residuals_.resize(cost_function_->num_residuals());
  std::vector<int> block_sizes = cost_function_->parameter_block_sizes();
  raw_jacobi_ = new double *[block_sizes.size()];
  jacobi_.resize(block_sizes.size());
  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
  {
    jacobi_[i].resize(cost_function_->num_residuals(), block_sizes[i]);
    raw_jacobi_[i] = jacobi_[i].data();
  }
  /// [residuals_] and [raw_jacobi_] is empty, and [raw_jacobi_] is equal to [jacobi_]
  cost_function_->Evaluate(parameter_blocks_.data(), residuals_.data(), raw_jacobi_);
  if (loss_function_)
  {
    double residual_scaling_, alpha_sq_norm_;
    double sq_norm, rho[3];
    sq_norm = residuals_.squaredNorm();
    loss_function_->Evaluate(sq_norm, rho);
    double sqrt_rho1_ = sqrt(rho[1]);
    if ((sq_norm == 0.0) || (rho[2] <= 0.0))
    {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    }
    else
    {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }
    for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); i++)
    {
      jacobi_[i] = sqrt_rho1_ * (jacobi_[i] - alpha_sq_norm_ * residuals_ * (residuals_.transpose() * jacobi_[i]));
    }
    residuals_ *= residual_scaling_;
  }
}
} // namespace factor