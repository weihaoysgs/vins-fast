//
// Created by weihao on 23-10-7.
//

#ifndef SRC_TEST_IMU_TOOL_HPP
#define SRC_TEST_IMU_TOOL_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include "Eigen/Core"
#include "Eigen/Dense"
class Param
{
public:
  Param();

  // time
  static const int imu_frequency = 200;
  static const int cam_frequency = 30;
  constexpr static const double imu_timestep = 1. / imu_frequency;
  constexpr static const double cam_timestep = 1. / cam_frequency;
  constexpr static const double t_start = 0.;
  constexpr static const double t_end = 20; //  20 s

  // noise
  constexpr static const double gyro_bias_sigma = 1.0e-5;
  constexpr static const double acc_bias_sigma = 0.0001;

  constexpr static const double gyro_noise_sigma = 0.015; // rad/s * 1/sqrt(hz)
  constexpr static const double acc_noise_sigma = 0.019;  //　m/(s^2) * 1/sqrt(hz)
};
struct MotionData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  Eigen::Matrix3d Rwb;
  Eigen::Vector3d twb;
  Eigen::Vector3d imu_acc;
  Eigen::Vector3d imu_gyro;

  Eigen::Vector3d imu_gyro_bias;
  Eigen::Vector3d imu_acc_bias;

  Eigen::Vector3d imu_velocity;
};

inline void LoadPose(std::string filename, std::vector<MotionData> &pose)
{
  std::ifstream f;
  f.open(filename.c_str());

  if (!f.is_open())
  {
    std::cerr << " can't open LoadFeatures file " << std::endl;
    return;
  }

  while (!f.eof())
  {
    std::string s;
    std::getline(f, s);

    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;

      MotionData data;
      double time;
      Eigen::Quaterniond q;
      Eigen::Vector3d t;
      Eigen::Vector3d gyro;
      Eigen::Vector3d acc;
      Eigen::Vector3d vel;

      ss >> time;
      ss >> q.w();
      ss >> q.x();
      ss >> q.y();
      ss >> q.z();
      ss >> t(0);
      ss >> t(1);
      ss >> t(2);
      ss >> vel(0);
      ss >> vel(1);
      ss >> vel(2);
      ss >> gyro(0);
      ss >> gyro(1);
      ss >> gyro(2);
      ss >> acc(0);
      ss >> acc(1);
      ss >> acc(2);

      data.timestamp = time;
      data.imu_gyro = gyro;
      data.imu_acc = acc;
      data.twb = t;
      data.imu_velocity = vel;
      data.Rwb = Eigen::Matrix3d(q);
      pose.push_back(data);
    }
  }
}

inline void ImuIntegration(std::vector<MotionData> &imudata, std::string dist)
{
  std::ofstream save_points;
  save_points.open(dist);
  double dt = Param::imu_timestep;

  Eigen::Vector3d Pwb = imudata.at(0).twb;            /// position :    from  imu measurements
  Eigen::Quaterniond Qwb(imudata.at(0).Rwb);    /// quaterniond:  from imu measurements
  Eigen::Vector3d Vw = imudata[0].imu_velocity;          /// velocity  :   from imu measurements
  Eigen::Vector3d gw(0, 0, -9.81);              /// ENU frame
  Eigen::Vector3d temp_a;
  Eigen::Vector3d theta;
  // std::cout << Qwb.coeffs().transpose() << "; " << Pwb.transpose() << "; " << Vw.transpose() << std::endl;
  for (int i = 1; i < imudata.size(); ++i)
  {
    // MotionData imupose = imudata[i];
    //
    // Eigen::Quaterniond dq;
    // Eigen::Vector3d dtheta_half = imupose.imu_gyro * dt / 2.0;
    // dq.w() = 1;
    // dq.x() = dtheta_half.x();
    // dq.y() = dtheta_half.y();
    // dq.z() = dtheta_half.z();
    // dq.normalize();
    //
    // Eigen::Vector3d acc_w = Qwb * (imupose.imu_acc) + gw; // aw = Rwb * ( acc_body - acc_bias ) + gw
    // Qwb = Qwb * dq;
    // Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
    // Vw = Vw + acc_w * dt;

    /// 中值积分
    MotionData imupose_ = imudata[i - 1];
    MotionData imupose = imudata[i];
    Eigen::Quaterniond dq;
    Eigen::Vector3d dtheta_half = (imupose_.imu_gyro + imupose.imu_gyro) * dt / 4.0;
    dq.w() = 1;
    dq.x() = dtheta_half.x();
    dq.y() = dtheta_half.y();
    dq.z() = dtheta_half.z();
    dq.normalize();
    Eigen::Vector3d acc_w = (Qwb * dq * (imupose.imu_acc) + gw + Qwb * (imupose_.imu_acc) + gw) /
                            2; /// 这里注意下一时刻对应的转换矩阵是Qwb * dq，而不是Qwb
    Qwb = Qwb * dq;
    Pwb = Pwb + Vw * dt + 0.5 * dt * dt * acc_w;
    Vw = Vw + acc_w * dt;
    //　按着imu postion, imu quaternion , cam postion, cam quaternion 的格式存储，由于没有cam，所以imu存了两次
    save_points << imupose.timestamp << " " << Qwb.w() << " " << Qwb.x() << " " << Qwb.y() << " " << Qwb.z() << " "
                << Pwb(0) << " " << Pwb(1) << " " << Pwb(2) << " " << Qwb.w() << " " << Qwb.x() << " " << Qwb.y() << " "
                << Qwb.z() << " " << Pwb(0) << " " << Pwb(1) << " " << Pwb(2) << " " << std::endl;

    // save_points.precision(9);
    // save_points << imupose.timestamp << " ";
    // save_points.precision(5);
    // save_points << Pwb(0) << " " << Pwb(1) << " " << Pwb(2) << " " << Qwb.x() << " " << Qwb.y() << " " << Qwb.z() << " "
    //             << Qwb.w() << std::endl;
  }

  std::cout << "Integration Success, Final State: \n";
  std::cout << "Q: " << Qwb.coeffs().transpose() << ";P: " << Pwb.transpose() << "; V:" << Vw.transpose() << std::endl;
}

#endif //SRC_TEST_IMU_TOOL_HPP
