//
// Created by weihao on 23-10-6.
//

#ifndef SRC_MARGINALIZATION_INFO_HPP
#define SRC_MARGINALIZATION_INFO_HPP

#include "factor/marginalization/marg_residual_block_info.hpp"
#include "factor/marginalization/marg_thread_struct.hpp"
#include <glog/logging.h>
#include <pthread.h>
#include <unordered_map>

namespace factor {

class MarginalizationInfo
{
public:
  MarginalizationInfo() = default;
  ~MarginalizationInfo();
  void AddResidualBlockInfo(MargResidualBlockInfo *residual_block_info);
  void PreMarginalize();
  void Marginalize();
  std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

  int LocalSize(int size) const { return size == 7 ? 6 : size; }
  bool valid_ = true;
  int m_, n_;
  const double eps_ = 1e-8;
  std::vector<int> keep_block_size_; //global size
  std::vector<int> keep_block_idx_;  //local size
  std::vector<double *> keep_block_data_;

  std::vector<MargResidualBlockInfo *> factors_;
  std::unordered_map<long, int> map_parameter_block_size_; /// global size
  std::unordered_map<long, int> map_parameter_block_idx_;  /// local size
  std::unordered_map<long, double *> map_parameter_block_data_;
  Eigen::MatrixXd linearized_jacobi_;
  Eigen::VectorXd linearized_residuals_;
};

} // namespace factor

#endif //SRC_MARGINALIZATION_INFO_HPP
