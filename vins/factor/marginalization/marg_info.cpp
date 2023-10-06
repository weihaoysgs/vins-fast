//
// Created by weihao on 23-10-6.
//
#include "factor/marginalization/marg_info.hpp"

namespace factor {

void MarginalizationInfo::AddResidualBlockInfo(MargResidualBlockInfo *residual_block_info)
{
  factors_.emplace_back(residual_block_info);
  std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks_;
  std::vector<int> parameter_block_sizes = residual_block_info->cost_function_->parameter_block_sizes();
  /// Traverse the parameter, map the parameter addr and global size
  for (int i = 0; i < parameter_blocks.size(); i++)
  {
    double *addr = parameter_blocks[i];
    int size = parameter_block_sizes[i];
    map_parameter_block_size_[reinterpret_cast<long>(addr)] = size;
  }
  /// Traverse the parameters that require marg in the residual block
  for (int i = 0; i < residual_block_info->drop_set_.size(); i++)
  {
    double *addr = parameter_blocks[residual_block_info->drop_set_[i]];
    map_parameter_block_idx_[reinterpret_cast<long>(addr)] = 0;
  }
}

void MarginalizationInfo::PreMarginalize()
{
  for (auto it : factors_)
  {
    /// compute jacobi and residual
    it->Evaluate();

    std::vector<int> block_size = it->cost_function_->parameter_block_sizes();
    for (int i = 0; i < block_size.size(); i++)
    {
      long addr = reinterpret_cast<long>(it->parameter_blocks_[i]);
      int size = block_size[i];
      assert(size == map_parameter_block_size_[addr]);
      if (map_parameter_block_data_.find(addr) == map_parameter_block_data_.end())
      {
        auto *data = new double[size];
        memcpy(data, it->parameter_blocks_[i], sizeof(double) * size);
        map_parameter_block_data_[addr] = data;
      }
    }
  }
}

void MarginalizationInfo::Marginalize()
{
  int pos = 0;
  for (auto &it : map_parameter_block_idx_)
  {
    it.second = pos;
    pos += LocalSize(map_parameter_block_size_[it.first]);
  }
  /// The dimension of the variable that needs to be marg out
  m_ = pos;
  /// The loop here plus the inner judgment actually traverses the variables to be retained.
  for (const auto &it : map_parameter_block_size_)
  {
    /// If this address is not found in the variable to be marg, it means that the
    /// parameter block corresponding to this address is a variable to be retained.
    if (map_parameter_block_idx_.find(it.first) == map_parameter_block_idx_.end())
    {
      /// add new map variable to [map_parameter_block_idx_],
      /// now the [map_parameter_block_idx_] restore the index of every parameter block
      map_parameter_block_idx_[it.first] = pos;
      pos += LocalSize(it.second);
    }
  }
  /// Dimensions of variables to be retained
  n_ = pos - m_;
  if (m_ == 0)
  {
    valid_ = false;
    LOG(ERROR) << "Unstable tracking...";
    return;
  }

  Eigen::MatrixXd A(pos, pos);
  Eigen::VectorXd b(pos);
  A.setZero();
  b.setZero();
  const int NUM_THREADS = 4;
  pthread_t t_threads[NUM_THREADS];
  ThreadsStruct threads_struct[NUM_THREADS];
  int i = 0;
  for (auto it : factors_)
  {
    threads_struct[i].sub_factors_.push_back(it);
    i++;
    i = i % NUM_THREADS;
  }
  for (int i = 0; i < NUM_THREADS; i++)
  {
    threads_struct[i].A_ = Eigen::MatrixXd::Zero(pos, pos);
    threads_struct[i].b_ = Eigen::VectorXd::Zero(pos);
    threads_struct[i].map_parameter_block_size_ = map_parameter_block_size_;
    threads_struct[i].map_parameter_block_idx_ = map_parameter_block_idx_;
    int ret = pthread_create(&t_threads[i], nullptr, ThreadsConstructA, (void *)&(threads_struct[i]));
    if (ret != 0)
    {
      LOG(WARNING) << "pthread_create error";
      break;
    }
  }
  for (int i = NUM_THREADS - 1; i >= 0; i--)
  {
    pthread_join(t_threads[i], nullptr);
    A += threads_struct[i].A_;
    b += threads_struct[i].b_;
  }

  /// 矩阵左上角
  Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m_, m_) + A.block(0, 0, m_, m_).transpose());
  /// 特征值分解
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

  /// 通过特征值分解求逆
  Eigen::MatrixXd Amm_inv =
      saes.eigenvectors() *
      Eigen::VectorXd((saes.eigenvalues().array() > eps_).select(saes.eigenvalues().array().inverse(), 0))
          .asDiagonal() *
      saes.eigenvectors().transpose();

  /// 得到每个部分，进行舒尔补
  Eigen::VectorXd bmm = b.segment(0, m_);
  Eigen::MatrixXd Amr = A.block(0, m_, m_, n_);
  Eigen::MatrixXd Arm = A.block(m_, 0, n_, m_);
  Eigen::MatrixXd Arr = A.block(m_, m_, n_, n_);
  Eigen::VectorXd brr = b.segment(m_, n_);
  /// 先验信息 Ax=b
  A = Arr - Arm * Amm_inv * Amr;
  b = brr - Arm * Amm_inv * bmm;

  /// 将 A 分解成 J^T * J
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps_).select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps_).select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  linearized_jacobi_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  linearized_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

MarginalizationInfo::~MarginalizationInfo()
{
  for (auto it = map_parameter_block_data_.begin(); it != map_parameter_block_data_.end(); ++it)
    delete it->second;
  for (int i = 0; i < (int)factors_.size(); i++)
  {
    delete[] factors_[i]->raw_jacobi_;
    delete factors_[i]->cost_function_;
    delete factors_[i];
  }
}
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
  std::vector<double *> keep_block_addr;
  keep_block_size_.clear();
  keep_block_idx_.clear();
  keep_block_data_.clear();

  /// 遍历所有的参数 <address : index in A matrix>
  for (const auto &it : map_parameter_block_idx_)
  {
    /// reserve 的参数（变量）
    if (it.second >= m_)
    {
      keep_block_size_.push_back(map_parameter_block_size_[it.first]);
      keep_block_idx_.push_back(map_parameter_block_idx_[it.first]);
      keep_block_data_.push_back(map_parameter_block_data_[it.first]);
      keep_block_addr.push_back(addr_shift[it.first]);
    }
  }
  // sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);
  /// 仍然保留的变量的地址
  return keep_block_addr;
}
} // namespace factor