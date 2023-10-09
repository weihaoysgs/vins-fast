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
const int IMU_FREQUENCY = 20;

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

void GenerateOptimationSimulateData()
{
  Eigen::Vector3d G{0.0, 0.0, 9.81};
  Eigen::Quaterniond Qi, Qj;
  Eigen::Vector3d Pi, Pj;

  int circle_time = 0;
  bool first_imu_flag = true;
  for (size_t i = 0; i < imu_data.size(); i = i + IMU_FREQUENCY)
  {
    circle_time++;
    double start_time = imu_data[i].timestamp_;
    double end_time = imu_data[i + IMU_FREQUENCY - 1].timestamp_;
    // int c_time = 0;
    ImuData first_imu = imu_data[i];
    Eigen::Vector3d ba, bg;
    ba.setZero(), bg.setZero();
    factor::IntegrationBase imu_pre_integ(first_imu.acc_, first_imu.gyro_, ba, bg);
    for (int j = i; j < i + IMU_FREQUENCY; j++)
    {
      // c_time++;
      ImuData imu = imu_data[j];
      double dt;
      if (first_imu_flag)
      {
        first_imu_flag = false;
        dt = 0;
      }
      else
      {
        dt = imu.timestamp_ - imu_data[j - 1].timestamp_;
      }
      imu_pre_integ.push_back(dt, imu.acc_, imu.gyro_);
    }
    int pose_start_index = FindStartIndexAccordingTimestamp(start_time, true);
    int pose_end_index = FindStartIndexAccordingTimestamp(end_time, true);
    if (pose_start_index < 0 || pose_end_index < 0)
      continue;
    GTPose gt_start_pose, gt_end_pose;
    double sum_dt = end_time - start_time;
    GTPose start_pose = gt_pose[pose_start_index];
    gt_start_pose = start_pose;
    GTPose end_pose = gt_pose[pose_end_index];
    gt_end_pose = end_pose;

    start_pose.P_ = start_pose.P_ + Eigen::Vector3d ::Random().normalized() * 1.3;
    end_pose.P_ = end_pose.P_ + Eigen::Vector3d ::Random().normalized() * 1.3;

    Eigen::Quaterniond Qi, Qj;
    Eigen::Vector3d Pi, Vi, Vj, Pj;
    Qi = start_pose.R_;
    Pi = start_pose.P_;
    Vi = start_pose.V_;
    Qj = end_pose.R_;
    Pj = end_pose.P_;
    Vj = end_pose.V_;
    Eigen::Vector3d p_err = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - imu_pre_integ.delta_p_;

    double param_pose_i[7], param_pose_j[7];
    double speed_bias_i[9], speed_bias_j[9];
    param_pose_i[0] = Pi.x(), param_pose_j[0] = Pj.x();
    param_pose_i[1] = Pi.y(), param_pose_j[1] = Pj.y();
    param_pose_i[2] = Pi.z(), param_pose_j[2] = Pj.z();
    param_pose_i[3] = Qi.x(), param_pose_j[3] = Qj.x();
    param_pose_i[4] = Qi.y(), param_pose_j[4] = Qj.y();
    param_pose_i[5] = Qi.z(), param_pose_j[5] = Qj.z();
    param_pose_i[6] = Qi.w(), param_pose_j[6] = Qj.w();
    speed_bias_i[0] = Vi.x(), speed_bias_j[0] = Vj.x();
    speed_bias_i[1] = Vi.y(), speed_bias_j[1] = Vj.y();
    speed_bias_i[2] = Vi.z(), speed_bias_j[2] = Vj.z();
    speed_bias_i[3] = 0, speed_bias_j[3] = 0;
    speed_bias_i[4] = 0, speed_bias_j[4] = 0;
    speed_bias_i[5] = 0, speed_bias_j[5] = 0;
    speed_bias_i[6] = 0, speed_bias_j[6] = 0;
    speed_bias_i[7] = 0, speed_bias_j[7] = 0;
    speed_bias_i[8] = 0, speed_bias_j[8] = 0;
    ceres::Problem problem;
    ceres::LocalParameterization *pose_i_local_param = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_pose_i, 7, pose_i_local_param);
    ceres::LocalParameterization *pose_j_local_param = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_pose_j, 7, pose_j_local_param);

    auto *imu_factor = new factor::IMUFactor(&imu_pre_integ);
    problem.AddResidualBlock(imu_factor, nullptr, param_pose_i, speed_bias_i, param_pose_j, speed_bias_j);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 7;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout.precision(10);

    Eigen::Vector3d baj, bgj;
    Pi.x() = param_pose_i[0], Pj.x() = param_pose_j[0];
    Pi.y() = param_pose_i[1], Pj.y() = param_pose_j[1];
    Pi.z() = param_pose_i[2], Pj.z() = param_pose_j[2];
    Qi.x() = param_pose_i[3], Qj.x() = param_pose_j[3];
    Qi.y() = param_pose_i[4], Qj.y() = param_pose_j[4];
    Qi.z() = param_pose_i[5], Qj.z() = param_pose_j[5];
    Qi.w() = param_pose_i[6], Qj.w() = param_pose_j[6];
    Vi.x() = speed_bias_i[0], Vj.x() = speed_bias_j[0];
    Vi.y() = speed_bias_i[1], Vj.y() = speed_bias_j[1];
    Vi.z() = speed_bias_i[2], Vj.z() = speed_bias_j[2];
    ba.x() = speed_bias_i[3], baj.x() = speed_bias_j[3];
    ba.y() = speed_bias_i[4], baj.y() = speed_bias_j[4];
    ba.z() = speed_bias_i[5], baj.z() = speed_bias_j[5];
    bg.x() = speed_bias_i[6], bgj.x() = speed_bias_j[6];
    bg.y() = speed_bias_i[7], bgj.y() = speed_bias_j[7];
    bg.z() = speed_bias_i[8], bgj.z() = speed_bias_j[8];
    // std::cout << "gt Pi: " << gt_start_pose.P_.transpose() <<"; before opt Pi(noise): " << Pi.transpose() << ", after opt Pi: " << param_pose_i[0] << ", " << param_pose_i[1]
    //           << ", " << param_pose_i[2] << std::endl;
    Eigen::Matrix3d dq_dbg = imu_pre_integ.jacobian_.block<3, 3>(O_R, O_BG);
    Eigen::Matrix3d dp_dbg = imu_pre_integ.jacobian_.block<3, 3>(O_P, O_BG);
    Eigen::Matrix3d dp_dba = imu_pre_integ.jacobian_.block<3, 3>(O_P, O_BA);

    Eigen::Vector3d dbg = bg - Eigen::Vector3d::Zero();
    Eigen::Vector3d dba = ba - Eigen::Vector3d::Zero();
    Eigen::Vector3d corrected_delta_p = imu_pre_integ.delta_p_ + dp_dba * dba + dp_dbg * dbg;
    Eigen::Vector3d p_err_opt =
        Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - imu_pre_integ.delta_p_;
    Eigen::Vector3d p_err_opt_dba_dbg =
        Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    std::cout.precision(7);

    std::cout << "P err: " << p_err.norm() << ", opt: " << p_err_opt.norm()
              << ", opt bg ba: " << p_err_opt_dba_dbg.norm() << "; " << p_err_opt.norm() - p_err_opt_dba_dbg.norm()
              << std::endl;
    // std::cout << "pose_start_index: " << pose_start_index << ", pose_end_index: " << pose_end_index << std::endl;
    // std::cout << "Integration: " << imu_pre_integ.delta_p_.transpose() << std::endl;
    // std::cout << "Ctime: " << c_time << std::endl;
    // std::cout << "start time: " << start_time << ", end time: " << end_time << ", dt: " << end_time - start_time
    //           << ", FREQUENCY: " << 1. / (end_time - start_time) << std::endl;
  }
  std::cout << "Circle time: " << circle_time << std::endl;
}