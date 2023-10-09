//
// Created by weihao on 23-10-8.
//

#ifndef DATASET_HPP
#define DATASET_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include <fstream>
#include <iostream>

struct ImuData
{
public:
  ImuData() = default;
  double timestamp_ = 0.;
  Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_ = Eigen::Vector3d ::Zero();
};

struct GTPose
{
public:
  GTPose() = default;
  double timestamp_ = 0.;
  Eigen::Vector3d P_ = Eigen::Vector3d ::Zero();
  Eigen::Quaterniond R_ = Eigen::Quaterniond ::Identity();
  Eigen::Vector3d V_ = Eigen::Vector3d ::Identity();
};

// timestamp, gx, gy, gz, ax, ay, az
inline void ReadImuRawDataset(const std::string &file_path, std::vector<ImuData> &imu_data)
{
  std::ifstream fin;
  fin.open(file_path.c_str());
  if (!fin.is_open())
  {
    std::cerr << " can't open LoadFeatures file " << std::endl;
    return;
  }
  while (!fin.eof())
  {
    std::string s;
    std::getline(fin, s);

    if (!s.empty() && s[0] != '#')
    {
      std::stringstream ss;
      ss << s;

      ImuData data;
      double time;

      Eigen::Vector3d gyro;
      Eigen::Vector3d acc;
      ss >> time;
      ss >> gyro(0);
      ss >> gyro(1);
      ss >> gyro(2);
      ss >> acc(0);
      ss >> acc(1);
      ss >> acc(2);
      data.acc_ = acc;
      data.gyro_ = gyro;
      data.timestamp_ = time / 1e9;
      imu_data.push_back(data);
    }
  }
  LOG(INFO) << "IMU Data size: " << imu_data.size() << std::endl;
  LOG(INFO) << "First IMU Data: " << imu_data[0].timestamp_ << ", " << imu_data[0].gyro_.transpose() << ", "
            << imu_data[0].acc_.transpose();
  LOG(INFO) << "End IMU Data: " << imu_data[imu_data.size() - 1].timestamp_ << ", "
            << imu_data[imu_data.size() - 1].gyro_.transpose() << ", "
            << imu_data[imu_data.size() - 1].acc_.transpose();
}

// #timestamp  p_RS_R_x [m]  p_RS_R_y [m]  p_RS_R_z [m]  q_RS_w []  q_RS_x []  q_RS_y []  q_RS_z []
// v_RS_R_x [m s^-1]  v_RS_R_y [m s^-1]  v_RS_R_z [m s^-1]  b_w_RS_S_x [rad s^-1]  b_w_RS_S_y [rad s^-1]
// b_w_RS_S_z [rad s^-1]  b_a_RS_S_x [m s^-2]  b_a_RS_S_y [m s^-2]  b_a_RS_S_z [m s^-2]
inline void ReadGroundTruthEurocPose(const std::string &pose_file_path, std::vector<GTPose> &gt_pose)
{
  std::ifstream fin;
  fin.open(pose_file_path);
  if (!fin.is_open())
  {
    LOG(FATAL) << "GT Pose File Open Failed";
  }
  while (!fin.eof())
  {
    std::string s;
    std::getline(fin, s);
    if (!s.empty()&& s[0] != '#')
    {
      std::stringstream ss;
      ss << s;

      GTPose data;
      double time;

      Eigen::Vector3d t,vel;
      Eigen::Quaterniond q;
      ss >> time;
      ss >> t(0);
      ss >> t(1);
      ss >> t(2);
      ss >> q.w();
      ss >> q.x();
      ss >> q.y();
      ss >> q.z();
      ss >> vel(0);
      ss >> vel(1);
      ss >> vel(2);
      data.timestamp_ = time;
      data.R_ = q;
      data.P_ = t;
      data.V_ = vel;
      gt_pose.push_back(data);
    }
  }
  LOG(INFO) << "Pose have :" << gt_pose.size();
  LOG(INFO) << "First Pose Data: " << gt_pose[0].timestamp_ << ", " << gt_pose[0].R_.coeffs().transpose() << ", "
            << gt_pose[0].P_.transpose() << ", vel: " << gt_pose[0].V_.transpose();
  LOG(INFO) << "End Pose Data: " << gt_pose[gt_pose.size() - 1].timestamp_ << ", "
            << gt_pose[gt_pose.size() - 1].R_.coeffs().transpose() << ", "
            << gt_pose[gt_pose.size() - 1].P_.transpose() << ", vel: " << gt_pose[gt_pose.size() - 1].V_.transpose();
}


#endif //DATASET_HPP
