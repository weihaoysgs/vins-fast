//
// Created by weihao on 23-10-7.
//
#include "factor/imu/imu_factor.hpp"

namespace factor {
bool IMUFactor::cout_imu_residual_ = false;
bool IMUFactor::Evaluate(const double *const *parameters, double *double_residuals, double **jacobians) const
{
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
  Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

  Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
  Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(double_residuals);
  residual = pre_integration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
  Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance_.inverse()).matrixL().transpose();
  // sqrt_info.setIdentity();
  residual = sqrt_info * residual;
  if (jacobians)
  {
    Eigen::Vector3d G = pre_integration_->G;
    double sum_dt = pre_integration_->sum_dt_;
    Eigen::Matrix3d dp_dba = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BG);
    Eigen::Matrix3d dq_dbg = pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG);
    Eigen::Matrix3d dv_dba = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BG);
    if (pre_integration_->jacobian_.maxCoeff() > 1e8 || pre_integration_->jacobian_.minCoeff() < -1e8)
    {
      LOG(WARNING) << "numerical unstable in preintegration";
    }

    if (jacobians[0])
    {
      /// 首先要知道 PVQ,Ba,Bg 哪几个受 Ri Pi 影响
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
      jacobian_pose_i.setZero();
      /// 公式 4.48a
      jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
      /// 公式 4.48d
      jacobian_pose_i.block<3, 3>(O_P, O_R) =
          common::Algorithm::SkewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
      Eigen::Quaterniond corrected_delta_q =
          pre_integration_->delta_q_ * common::Algorithm::DeltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
      jacobian_pose_i.block<3, 3>(O_R, O_R) =
          -(common::Algorithm::Qleft(Qj.inverse() * Qi) * common::Algorithm::Qright(corrected_delta_q))
               .bottomRightCorner<3, 3>();
      /// 公式 4.47
      jacobian_pose_i.block<3, 3>(O_V, O_R) = common::Algorithm::SkewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
      jacobian_pose_i = sqrt_info * jacobian_pose_i;
      if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
      {
        LOG(WARNING) << "numerical unstable in preintegration";
      }
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
      jacobian_speedbias_i.setZero();
      jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
      jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
      jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
      jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
          -common::Algorithm::Qleft(Qj.inverse() * Qi * pre_integration_->delta_q_).bottomRightCorner<3, 3>() * dq_dbg;
      jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
      jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
      jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
      jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
      jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
      jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
      jacobian_pose_j.setZero();
      jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
      Eigen::Quaterniond corrected_delta_q =
          pre_integration_->delta_q_ * common::Algorithm::DeltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
      jacobian_pose_j.block<3, 3>(O_R, O_R) =
          common::Algorithm::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
      jacobian_pose_j = sqrt_info * jacobian_pose_j;
    }
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
      jacobian_speedbias_j.setZero();
      jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
      jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
      jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
      jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
    }
  }
  // if (cout_imu_residual_)
  // {
  //   LOG(INFO) << "IMU Residual: " << residual.transpose() << ", normal: " << residual.norm();
  // }
  return true;
}

void IMUFactor::ComputeResidual(const std::vector<const double *> parameters) const
{
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
  Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

  Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
  Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);
  Eigen::Matrix<double, 15, 1> residual;
  residual = pre_integration_->Evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
  if (cout_imu_residual_)
  {
    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance_.inverse()).matrixL().transpose();
    LOG(INFO) << "IMU Residual: " << residual.transpose() << "; normal: " << residual.norm()
              << "; sqrt residual: " << (sqrt_info * residual).norm() << std::endl;
  }
}
} // namespace factor