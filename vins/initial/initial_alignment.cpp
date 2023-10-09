//
// Created by weihao on 23-10-9.
//
#include "initial/initial_alignment.hpp"
namespace initial {

void ImageImuFrame::SolveGyroscopeBias(std::map<double, ImageImuFrame> &all_image_frame, Eigen::Vector3d *Bgs,
                                       int WINDOW_SIZE)
{
  Eigen::Matrix3d A;
  Eigen::Vector3d b;
  Eigen::Vector3d delta_bg;
  A.setZero();
  b.setZero();
  std::map<double, ImageImuFrame>::iterator frame_i;
  std::map<double, ImageImuFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
  {
    frame_j = next(frame_i);
    Eigen::MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    Eigen::VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij(frame_i->second.R_.transpose() * frame_j->second.R_);
    tmp_A = frame_j->second.pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG);
    tmp_b = 2 * (frame_j->second.pre_integration_->delta_q_.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  delta_bg = A.ldlt().solve(b);

  for (int i = 0; i <= WINDOW_SIZE; i++)
    Bgs[i] += delta_bg;

  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
  {
    frame_j = next(frame_i);
    frame_j->second.pre_integration_->Repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
  }
}
} // namespace initial
