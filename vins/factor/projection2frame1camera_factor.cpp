#include "factor/projection2frame1camera_factor.hpp"

namespace factor {
Eigen::Matrix2d ProjectionTwoFrameOneCamFactor::sqrt_info_;
ProjectionTwoFrameOneCamFactor::ProjectionTwoFrameOneCamFactor(const Eigen::Vector3d &pts_i,
                                                               const Eigen::Vector3d &pts_j,
                                                               const Eigen::Vector2d &vel_i,
                                                               const Eigen::Vector2d &vel_j, const double td_i,
                                                               const double td_j)
  : pts_i_(pts_i), pts_j_(pts_j), td_i_(td_i), td_j_(td_j)
{
  velocity_i_.x() = vel_i.x();
  velocity_i_.y() = vel_i.y();
  velocity_i_.z() = 0;
  velocity_j_.x() = vel_j.x();
  velocity_j_.y() = vel_j.y();
  velocity_j_.z() = 0;
}

bool ProjectionTwoFrameOneCamFactor::Evaluate(const double *const *parameters, double *residuals,
                                              double **jacobians) const
{
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  double inv_dep_i = parameters[3][0];
  double td = parameters[4][0];

  Eigen::Vector3d pts_i_td = pts_i_, pts_j_td = pts_j_;

  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
  // std::cout << residual.norm() << "; ";
  // if (residual.norm() > 0.0010)
  // {
  //   sqrt_info_.setZero();
  // }
  residual = sqrt_info_ * residual;

  if (jacobians)
  {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    /// the error derivative to point, middle variable
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
    reduce = sqrt_info_ * reduce;

    /// Pose_i:{R_i,T_i}
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobi_pose_i(jacobians[0]);
      Eigen::Matrix<double, 3, 6> jacobi_i;
      jacobi_i.leftCols<3>() = ric.transpose() * Rj.transpose();
      jacobi_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -common::Algorithm::SkewSymmetric(pts_imu_i);

      jacobi_pose_i.leftCols<6>() = reduce * jacobi_i;
      jacobi_pose_i.rightCols<1>().setZero();

      {
        // CheckPoseJacobi(jacobi_pose_i, Ri, Pi, Qj.toRotationMatrix(), Pj,
        //                 pts_i_td, pts_j_td, inv_dep_i, qic, tic, residual);
      }
    }

    /// Pose_j:{R_j,T_j}
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobi_pose_j(jacobians[1]);

      Eigen::Matrix<double, 3, 6> jacobi_j;
      jacobi_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
      jacobi_j.rightCols<3>() = ric.transpose() * common::Algorithm::SkewSymmetric(pts_imu_j);

      jacobi_pose_j.leftCols<6>() = reduce * jacobi_j;
      jacobi_pose_j.rightCols<1>().setZero();
    }

    /// ExParam:{R_ic,T_ic}
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobi_ex_pose(jacobians[2]);
      Eigen::Matrix<double, 3, 6> jacobi_ex;
      jacobi_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
      Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
      jacobi_ex.rightCols<3>() =
          -tmp_r * common::Algorithm::SkewSymmetric(pts_camera_i) +
          common::Algorithm::SkewSymmetric(tmp_r * pts_camera_i) +
          common::Algorithm::SkewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
      jacobi_ex_pose.leftCols<6>() = reduce * jacobi_ex;
      jacobi_ex_pose.rightCols<1>().setZero();
    }

    /// feature:{double depth}
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Vector2d> jacobi_feature(jacobians[3]);
      jacobi_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
    }

    /// td:{double td}
    if (jacobians[4])
    {
      Eigen::Map<Eigen::Vector2d> jacobi_td(jacobians[4]);
      jacobi_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i_ / inv_dep_i * -1.0 +
                  sqrt_info_ * velocity_j_.head(2);
    }
  }
  return true;
}

void ProjectionTwoFrameOneCamFactor::CheckPoseJacobi(const Eigen::Matrix<double, 2, 7> &analysis_jacobi,
                                                     const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi,
                                                     const Eigen::Matrix3d &Qj, const Eigen::Vector3d &Pj,
                                                     const Eigen::Vector3d &pts_i_td, const Eigen::Vector3d &pts_j_td,
                                                     const double inv_dep_i, const Eigen::Quaterniond &qic,
                                                     const Eigen::Vector3d &tic, const Eigen::Vector2d &residual) const
{
  Eigen::Matrix<double, 2, 6> numeric_jacobi = Eigen::Matrix<double, 2, 6>::Zero();
  const double eps = 1e-6;
  Eigen::Quaterniond q_temp;
  Eigen::Vector3d t_temp;
  for (int i = 0; i < 6; i++)
  {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(i % 3) = eps;
    if (i < 3)
    {
      q_temp = Eigen::Quaterniond(Ri);
      t_temp = Pi + delta;
    }
    /// 对旋转进行扰动
    else
    {
      Eigen::Matrix<double, 3, 1> half_theta = delta;
      half_theta /= static_cast<double>(2.0);
      Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(), half_theta.z());
      q_temp = (Ri * dq.toRotationMatrix());
      t_temp = Pi;
    }

    Eigen::Vector3d _pts_camera_i = pts_i_td / inv_dep_i;   /// 得到 i 时刻相机系下的坐标
    Eigen::Vector3d _pts_imu_i = qic * _pts_camera_i + tic; /// 得到 IMU 系下的坐标
    Eigen::Vector3d _pts_w = q_temp * _pts_imu_i + t_temp;  /// 得到世界系下的坐标
    Eigen::Vector3d _pts_imu_j = Qj.inverse() * (_pts_w - Pj); /// 根据 j 时刻的位姿将坐标点投影到 j 时刻的 IMU 坐标系下
    Eigen::Vector3d _pts_camera_j = qic.inverse() * (_pts_imu_j - tic); /// 得到 j 时刻相机系下的坐标
    Eigen::Vector2d _residual;

    double _dep_j = _pts_camera_j.z();                                   /// 得到 j 时刻的相机系下的深度值
    _residual = (_pts_camera_j / _dep_j).head<2>() - pts_j_td.head<2>(); /// 计算归一化坐标的残差
    _residual = sqrt_info_ * _residual;
    numeric_jacobi.col(i) = (_residual - residual) / eps;
  }
  std::cout << "analysis jacobi:\n" << analysis_jacobi << std::endl;
  std::cout << "numeric jacobi:\n" << numeric_jacobi << std::endl;
}
} // namespace factor
