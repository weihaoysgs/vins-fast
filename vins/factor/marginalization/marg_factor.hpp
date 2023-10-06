//
// Created by weihao on 23-10-6.
//

#ifndef MARGINALIZATION_FACTOR_HPP
#define MARGINALIZATION_FACTOR_HPP

#include "factor/marginalization/marg_info.hpp"
#include "common/algorithm.hpp"

namespace factor {
class MarginalizationFactor : public ceres::CostFunction
{
  MarginalizationFactor(MarginalizationInfo *marginalization_info);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobi) const;
  MarginalizationInfo *marginalization_info_;
};
} // namespace factor

#endif // MARGINALIZATION_FACTOR_HPP
