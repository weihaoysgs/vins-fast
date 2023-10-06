//
// Created by weihao on 23-10-6.
//

#ifndef MARG_THREAD_STRUCT_HPP
#define MARG_THREAD_STRUCT_HPP

#include "factor/marginalization/marg_residual_block_info.hpp"
#include <unordered_map>

namespace factor {
struct ThreadsStruct
{
  std::vector<MargResidualBlockInfo *> sub_factors_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  std::unordered_map<long, int> map_parameter_block_size_; /// global size
  std::unordered_map<long, int> map_parameter_block_idx_;  /// local size
};

inline void *ThreadsConstructA(void *threads_struct)
{
  ThreadsStruct *p = ((ThreadsStruct *)threads_struct);
  for (auto it : p->sub_factors_)
  {
    for (int i = 0; i < static_cast<int>(it->parameter_blocks_.size()); i++)
    {
      int idx_i = p->map_parameter_block_idx_[reinterpret_cast<long>(it->parameter_blocks_[i])];
      int size_i = p->map_parameter_block_size_[reinterpret_cast<long>(it->parameter_blocks_[i])];
      if (size_i == 7)
        size_i = 6;
      Eigen::MatrixXd jacobi_i = it->jacobi_[i].leftCols(size_i);
      for (int j = i; j < static_cast<int>(it->parameter_blocks_.size()); j++)
      {
        int idx_j = p->map_parameter_block_idx_[reinterpret_cast<long>(it->parameter_blocks_[j])];
        int size_j = p->map_parameter_block_size_[reinterpret_cast<long>(it->parameter_blocks_[j])];
        if (size_j == 7)
          size_j = 6;
        Eigen::MatrixXd jacobi_j = it->jacobi_[j].leftCols(size_j);
        if (i == j)
          p->A_.block(idx_i, idx_j, size_i, size_j) += jacobi_i.transpose() * jacobi_j;
        else
        {
          p->A_.block(idx_i, idx_j, size_i, size_j) += jacobi_i.transpose() * jacobi_j;
          p->A_.block(idx_j, idx_i, size_j, size_i) = p->A_.block(idx_i, idx_j, size_i, size_j).transpose();
        }
      }
      p->b_.segment(idx_i, size_i) += jacobi_i.transpose() * it->residuals_;
    }
  }
  return threads_struct;
}

} // namespace factor

#endif // MARG_THREAD_STRUCT_HPP
