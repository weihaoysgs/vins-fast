//
// Created by weihao on 23-10-15.
//

#ifndef SRC_GLOBAL_POSE_FACTOR_HPP
#define SRC_GLOBAL_POSE_FACTOR_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace factor{
template <typename T> inline void QuaternionInverse(const T q[4], T q_inverse[4])
{
  q_inverse[0] = q[0];
  q_inverse[1] = -q[1];
  q_inverse[2] = -q[2];
  q_inverse[3] = -q[3];
};

struct RelativeRTError
{
  RelativeRTError(double t_x, double t_y, double t_z, double q_w, double q_x, double q_y, double q_z, double t_var,
                  double q_var)
    : t_x(t_x), t_y(t_y), t_z(t_z), q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z), t_var(t_var), q_var(q_var)
  {
  }

  template <typename T> bool operator()(const T *ti, const T *tj, T *residuals) const
  {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T i_q_w[4];
    T qi[4];
    qi[0] = ti[6];
    qi[1] = ti[3];
    qi[2] = ti[4];
    qi[3] = ti[5];
    QuaternionInverse(qi, i_q_w);

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x));// / T(t_var);
    residuals[1] = (t_i_ij[1] - T(t_y)) ;/// T(t_var);
    residuals[2] = (t_i_ij[2] - T(t_z)) ;/// T(t_var);

    T relative_q[4];
    relative_q[0] = T(q_w);
    relative_q[1] = T(q_x);
    relative_q[2] = T(q_y);
    relative_q[3] = T(q_z);

    T q_i_j[4];
    T qj[4];
    qj[0] = tj[6];
    qj[1] = tj[3];
    qj[2] = tj[4];
    qj[3] = tj[5];
    ceres::QuaternionProduct(i_q_w, qj, q_i_j);

    T relative_q_inv[4];
    /// relative_q Qij; relative_q_inv Qji
    QuaternionInverse(relative_q, relative_q_inv);

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);


    residuals[3] = T(2) * error_q[1] / T(q_var);
    residuals[4] = T(2) * error_q[2] / T(q_var);
    residuals[5] = T(2) * error_q[3] / T(q_var);

    residuals[0] *= T(INFO);
    residuals[1] *= T(INFO);
    residuals[2] *= T(INFO);
    residuals[3] *= T(0);
    residuals[4] *= T(0);
    residuals[5] *= T(0);

    return true;
  }

  static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z, const double q_w,
                                     const double q_x, const double q_y, const double q_z, const double t_var,
                                     const double q_var)
  {
    return (new ceres::AutoDiffCostFunction<RelativeRTError, 6, 7, 7>(
        new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
  }

  double t_x, t_y, t_z, t_norm;
  double q_w, q_x, q_y, q_z;
  double t_var, q_var;
  const double INFO = 5000.0;
};
} // namespace factor

#endif //SRC_GLOBAL_POSE_FACTOR_HPP
