#include "factor/projection2frame2camera_factor.hpp"

namespace factor {
Eigen::Matrix2d ProjectionTwoFrameTwoCamFactor::sqrt_info_;
bool ProjectionTwoFrameTwoCamFactor::cout_residual_;
ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(const Eigen::Vector3d &_pts_i,
                                                               const Eigen::Vector3d &_pts_j,
                                                               const Eigen::Vector2d &_velocity_i,
                                                               const Eigen::Vector2d &_velocity_j, const double _td_i,
                                                               const double _td_j)
  : pts_i_(_pts_i), pts_j_(_pts_j), td_i_(_td_i), td_j_(_td_j)
{
  velocity_i_.x() = _velocity_i.x();
  velocity_i_.y() = _velocity_i.y();
  velocity_i_.z() = 0;
  velocity_j_.x() = _velocity_j.x();
  velocity_j_.y() = _velocity_j.y();
  velocity_j_.z() = 0;
}

bool ProjectionTwoFrameTwoCamFactor::Evaluate(const double *const *parameters, double *residuals,
                                              double **jacobians) const
{
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

  double inv_dep_i = parameters[4][0];

  double td = parameters[5][0];

  Eigen::Vector3d pts_i_td = pts_i_, pts_j_td = pts_j_;
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
  residual = sqrt_info_ * residual;
  if (jacobians)
  {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

      Eigen::Matrix<double, 3, 6> jaco_i;
      jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
      jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -common::Algorithm::SkewSymmetric(pts_imu_i);

      jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
      jacobian_pose_i.rightCols<1>().setZero();
    }

    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

      Eigen::Matrix<double, 3, 6> jaco_j;
      jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
      jaco_j.rightCols<3>() = ric2.transpose() * common::Algorithm::SkewSymmetric(pts_imu_j);

      jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
      jacobian_pose_j.rightCols<1>().setZero();
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri;
      jaco_ex.rightCols<3>() =
          ric2.transpose() * Rj.transpose() * Ri * ric * -common::Algorithm::SkewSymmetric(pts_camera_i);
      jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose.rightCols<1>().setZero();
    }
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[3]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = -ric2.transpose();
      jaco_ex.rightCols<3>() = common::Algorithm::SkewSymmetric(pts_camera_j);
      jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose1.rightCols<1>().setZero();
    }
    if (jacobians[4])
    {
      Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[4]);
      jacobian_feature =
          reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
    }
    if (jacobians[5])
    {
      Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[5]);
      jacobian_td = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i_ / inv_dep_i * -1.0 +
                    sqrt_info_ * velocity_j_.head(2);
    }
  }
  return true;
}

void ProjectionTwoFrameTwoCamFactor::ComputeResidual(const std::vector<const double *> parameters) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

  double inv_dep_i = parameters[4][0];

  double td = parameters[5][0];

  Eigen::Vector3d pts_i_td = pts_i_, pts_j_td = pts_j_;
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
  double dep_j = pts_camera_j.z();
  Eigen::Vector2d residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
  if(cout_residual_)
  {
    std::cout.precision(7);
    LOG(INFO) << "TwoFrameTwoCam Residual: " << residual.transpose() << "; normal: " << residual.norm()
              << "; sqrt residual: " << (sqrt_info_ * residual).norm() << std::endl;
  }
}
} // namespace factor
