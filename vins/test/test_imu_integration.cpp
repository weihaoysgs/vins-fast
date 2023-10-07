#include "factor/imu/imu_integration_base.hpp"
#include <gflags/gflags.h>

DEFINE_string(config_file_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/config/euroc/euroc_stero.yaml", "vins yaml config file path");

int main(int argc, char **argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  common::Setting::getSingleton()->InitParamSetting(fLS::FLAGS_config_file_path);

  Eigen::Vector3d acc_0, gyr_0, ba, bg;
  factor::IntegrationBase imu_integration(acc_0, gyr_0, ba, bg);
  return 0;
}