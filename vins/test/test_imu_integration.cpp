#include "factor/imu/imu_integration_base.hpp"
#include <gflags/gflags.h>
#include "test/imu/test_imu_tool.hpp"

DEFINE_string(config_file_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/config/euroc/euroc_stero.yaml",
              "vins yaml config file path");

DEFINE_string(imu_raw_simulate_data_path,
              "/home/weihao/codespace/whvio_ws/src/vins-fast/scripts/imu_integ/imu_raw_pose.txt",
              "vins yaml config file path");
DEFINE_string(imu_raw_simulate_noise_data_path,
              "/home/weihao/codespace/whvio_ws/src/vins-fast/scripts/imu_integ/imu_raw_noise_pose.txt",
              "vins yaml config file path");

DEFINE_string(imu_raw_simulate_data_integ_result_path,
              "/home/weihao/codespace/whvio_ws/src/vins-fast/scripts/imu_integ/gt_integration_result.txt",
              "vins yaml config file path");
DEFINE_string(imu_raw_simulate_noise_data_integ_result_path,
              "/home/weihao/codespace/whvio_ws/src/vins-fast/scripts/imu_integ/noise_integration_result.txt",
              "vins yaml config file path");

void EvaluateVinsIntegrationBase(std::vector<MotionData> &imu_data, const std::string &result_save_path);

int main(int argc, char **argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  common::Setting::getSingleton()->InitParamSetting(fLS::FLAGS_config_file_path);

  std::vector<MotionData> imu_raw_data, imu_raw_noise_data;
  LoadPose(fLS::FLAGS_imu_raw_simulate_data_path, imu_raw_data);
  LoadPose(fLS::FLAGS_imu_raw_simulate_noise_data_path, imu_raw_noise_data);
  LOG(INFO) << "data size: " << imu_raw_data.size() << ";" << imu_raw_noise_data.size();
  ImuIntegration(imu_raw_data, fLS::FLAGS_imu_raw_simulate_data_integ_result_path);
  // ImuIntegration(imu_raw_noise_data, fLS::FLAGS_imu_raw_simulate_noise_data_integ_result_path);

  EvaluateVinsIntegrationBase(imu_raw_data, "");
  // EvaluateVinsIntegrationBase(imu_raw_noise_data, "");
  return 0;
}

void EvaluateVinsIntegrationBase(std::vector<MotionData> &imu_data, const std::string &result_save_path)
{
  Eigen::Vector3d Pwb = imu_data.at(0).twb;
  Eigen::Quaterniond Qwb(imu_data.at(0).Rwb);
  Eigen::Vector3d Vw = imu_data[0].imu_velocity;
  std::cout << "Init P: " << Pwb.transpose() << std::endl;
  std::cout << "Init R: " << Qwb.coeffs().transpose() << std::endl;
  std::cout << "Init V: " << Vw.transpose() << std::endl;
  Eigen::Vector3d acc_0 = imu_data[0].imu_acc, gyr_0 = imu_data[0].imu_gyro, ba, bg;
  ba.setZero(), bg.setZero();
  factor::IntegrationBase imu_pre_integration(acc_0, gyr_0, ba, bg);
  double dt;
  for (int i = 1; i < imu_data.size(); i++)
  {
    imu_pre_integration.push_back(1.0 / Param::imu_frequency, imu_data[i].imu_acc, imu_data[i].imu_gyro);
    // if (i == 100)
    // {
    //   std::cout << "cc:" << imu_pre_integration.delta_p_.transpose() << std::endl;
    // }
  }
  double sum_dt = imu_pre_integration.sum_dt_;
  Eigen::Vector3d Pj =
      Qwb * imu_pre_integration.delta_p_ + Pwb + Vw * sum_dt - 0.5 * Eigen::Vector3d(0, 0, 9.81) * sum_dt * sum_dt;
  Eigen::Quaterniond Qj = (Qwb * imu_pre_integration.delta_q_);
  Eigen::Vector3d Vj = Qwb * imu_pre_integration.delta_v_ + Vw - Eigen::Vector3d(0, 0, 9.81) * sum_dt;
  std::cout << "Pj: " << Pj.transpose() << std::endl;
  std::cout << "Qj: " << Qj.coeffs().transpose() << std::endl;
  std::cout << "Vj: " << Vj.transpose() << std::endl;

  Eigen::Matrix<double, 15, 1> residual = imu_pre_integration.Evaluate(Pwb, Qwb, Vw, ba, bg, Pj, Qj, Vj, ba, bg);
  std::cout << "residual: " << residual.transpose() << std::endl;
}