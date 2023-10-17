//
// Created by weihao on 23-10-8.
//

#include "factor/imu/imu_factor.hpp"
#include "test/ins/dataset.hpp"
#include "factor/pose_local_parameterization.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>
DEFINE_string(config_file_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/config/euroc/euroc_stero.yaml",
              "vins yaml config file path");

DEFINE_string(imu_data_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/test/ins/dataset/MH_05_Imu.txt",
              "imu data path");
DEFINE_string(gt_pose_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/test/ins/dataset/MH_05_Gt_Pose.txt",
              "gt pose path");
const int IMU_FREQUENCY = 5;

std::vector<ImuData> imu_data;
std::vector<GTPose> gt_pose;

void GenerateOptimationSimulateData();

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_imu_factor");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);
  common::Setting::getSingleton()->InitParamSetting(fLS::FLAGS_config_file_path);

  ReadImuRawDataset(fLS::FLAGS_imu_data_path, imu_data);
  ReadGroundTruthEurocPose(fLS::FLAGS_gt_pose_path, gt_pose);
  GenerateOptimationSimulateData();
  return 0;
}

int FindStartIndexAccordingTimestamp(double timestamp, bool is_imu)
{
  for (int i = 0; i < gt_pose.size(); i++)
  {
    if (gt_pose[i].timestamp_ >= timestamp)
      return i;
  }
  return -1;
}

std::vector<ImuData> getImuInterval(double start_time, double end_time)
{
  std::vector<ImuData> ret_imu_data;
  for (int i = 0; i < imu_data.size(); i++)
  {
    ImuData data = imu_data[i];
    if (data.timestamp_ >= start_time && data.timestamp_ <= end_time)
      ret_imu_data.push_back(data);
  }
  return ret_imu_data;
}

void IntegrationIMU(factor::IntegrationBase &imu_pre_inter, std::vector<ImuData> &imu) { }

void GenerateOptimationSimulateData()
{
  Eigen::Vector3d G{0.0, 0.0, 9.81};
  Eigen::Quaterniond last_Qi, this_Qj;
  Eigen::Vector3d last_Pi, this_Pj;
  Eigen::Vector3d ba, bg;
  ba.setZero(), bg.setZero();

  int circle_time = 0;
  bool first_imu_flag = true;
  bool first_pose_set = false;
  for (size_t i = 0; i < gt_pose.size(); i = i + IMU_FREQUENCY)
  {
    last_Qi = gt_pose[i].R_;
    last_Pi = gt_pose[i].P_;
    this_Qj = gt_pose[i + IMU_FREQUENCY - 1].R_;
    this_Pj = gt_pose[i + IMU_FREQUENCY - 1].P_;

    double start_time = imu_data[i].timestamp_;
    double end_time = imu_data[i + IMU_FREQUENCY - 1].timestamp_;
    std::vector<ImuData> interval_imu_data = getImuInterval(start_time, end_time);
    ImuData first_imu = interval_imu_data[0];
    factor::IntegrationBase imu_pre_integ(first_imu.acc_, first_imu.gyro_, ba, bg);
    for (int j = 1; j < interval_imu_data.size(); j++)
    {
      double dt = interval_imu_data[j].timestamp_ - interval_imu_data[j - 1].timestamp_;
      imu_pre_integ.push_back(dt, interval_imu_data[j].acc_, interval_imu_data[j].gyro_);
    }

    Eigen::Vector3d gt_dp = this_Pj - last_Pi;
    std::cout << "imu dp: " << imu_pre_integ.delta_p_.transpose() << "; gt dp: " << gt_dp.transpose() << std::endl;

    std::cout.precision(10);
    // std::cout << "img start time: " << start_time << "; img end time: " << end_time
    //           << "; size: " << interval_imu_data.size() << "; start imu time: " << interval_imu_data[0].timestamp_
    //           << "; end imu data time: " << interval_imu_data[interval_imu_data.size() - 1].timestamp_ << std::endl;
  }
  std::cout << "Circle time: " << circle_time << std::endl;
}