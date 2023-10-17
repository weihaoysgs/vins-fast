//
// Created by weihao on 23-10-8.
//

#include "factor/imu/imu_factor.hpp"
#include "test/ins/dataset.hpp"
#include "test/ins/global_pose_factor.hpp"
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



void GenerateOptimationSimulateData(std::vector<ImuData>&imu_data, std::vector<GTPose>&gt_pose);

int main(int argc, char **argv)
{
  google::InitGoogleLogging("test_imu_factor");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);
  common::Setting::getSingleton()->InitParamSetting(fLS::FLAGS_config_file_path);
  std::vector<ImuData> imu_data;
  std::vector<GTPose> gt_pose;
  ReadImuRawDataset(fLS::FLAGS_imu_data_path, imu_data);
  ReadGroundTruthEurocPose(fLS::FLAGS_gt_pose_path, gt_pose);
  GenerateOptimationSimulateData(imu_data, gt_pose);
  return 0;
}

int FindStartIndexAccordingTimestamp(double timestamp, bool is_imu, std::vector<ImuData>&imu_data, std::vector<GTPose>&gt_pose)
{
  for (int i = 0; i < gt_pose.size(); i++)
  {
    if (gt_pose[i].timestamp_ >= timestamp)
    {
      double pose_time = gt_pose[i].timestamp_;
      double err = (pose_time - timestamp) * (pose_time - timestamp);
      if (err > 0.1)
        return -1;
      return i;
    }
  }
  return -1;
}

void GenerateOptimationSimulateData(std::vector<ImuData>&imu_data, std::vector<GTPose>&gt_pose)
{
  Eigen::Vector3d G{0.0, 0.0, 9.81};
  Eigen::Quaterniond Qi, Qj;
  Qj.setIdentity(), Qi.setIdentity();
  Eigen::Vector3d Pi, Vi, Vj, Pj;
  Pi.setZero(), Vi.setZero(), Vj.setZero(), Pj.setZero();
  Eigen::Vector3d ba, bg;
  Eigen::Vector3d baj, bgj;
  ba.setZero(), bg.setZero();
  GTPose init_pose;

  std::ofstream result_traj("/home/weihao/codespace/whvio_ws/output/ins.txt");


  int circle_time = 0;
  bool first_imu_flag = true;
  bool first_pose_set = false;
  for (size_t i = 0; i < imu_data.size(); i = i + IMU_FREQUENCY)
  {
    circle_time++;
    double start_time = imu_data[i].timestamp_;
    double end_time = imu_data[i + IMU_FREQUENCY - 1].timestamp_;
    int pose_start_index = FindStartIndexAccordingTimestamp(start_time, true, imu_data, gt_pose);
    int pose_end_index = FindStartIndexAccordingTimestamp(end_time, true, imu_data, gt_pose);
    if (pose_start_index < 0 || pose_end_index < 0)
      continue;
    assert(std::abs(gt_pose[pose_start_index].timestamp_ - start_time) < 0.5);
    assert(std::abs(gt_pose[pose_end_index].timestamp_ - end_time) < 0.5);

    // GTPose gt_start_pose, gt_end_pose;
    GTPose start_pose = gt_pose[pose_start_index];
    // gt_start_pose = start_pose;
    GTPose end_pose = gt_pose[pose_end_index];
    // gt_end_pose = end_pose;
    // int c_time = 0;
    init_pose = start_pose;
    if (!first_pose_set)
    {
      first_pose_set = true;
      Qi = init_pose.R_;
      Pi = init_pose.P_;
      Vi = init_pose.V_;
    }
    else
    {
      Qi = Qj;
      Pi = Pj;
      Vi = Vj;
    }
    ImuData first_imu = imu_data[i];
    factor::IntegrationBase imu_pre_integ(first_imu.acc_, first_imu.gyro_, ba, bg);
    Eigen::Quaterniond Q_temp = Qi;
    Eigen::Vector3d P_temp = Pi,V_temp = Vi;
    double sum_dt = 0;// = end_time - start_time;

    for (int j = i; j <= i + IMU_FREQUENCY; j++)
    {
      // c_time++;
      ImuData imu = imu_data[j];
      Eigen::Vector3d acc_0, acc_1, gyr_0, gyr_1;
      double dt;
      if (first_imu_flag)
      {
        first_imu_flag = false;
        dt = 0;
        acc_0 = imu_data[j].acc_;
        acc_1 = imu_data[j].acc_;
        gyr_0 = imu_data[j].gyro_;
        gyr_1 = imu_data[j].gyro_;
      }
      else
      {
        acc_0 = imu_data[j - 1].acc_;
        acc_1 = imu_data[j].acc_;
        gyr_0 = imu_data[j - 1].gyro_;
        gyr_1 = imu_data[j].gyro_;
        dt = imu.timestamp_ - imu_data[j - 1].timestamp_;
      }
      imu_pre_integ.push_back(dt, imu.acc_, imu.gyro_);
      sum_dt+=dt;
      Eigen::Vector3d un_acc_0 = Q_temp * (acc_0 - ba) - G; /// a^' = R^T * (a-g)
      Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - bg;
      Q_temp = Q_temp * common::Algorithm::DeltaQ(un_gyr * dt).toRotationMatrix();
      Eigen::Vector3d un_acc_1 = Q_temp * (acc_1 - ba) - G;
      Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
      P_temp += dt * V_temp + 0.5 * dt * dt * un_acc;
      V_temp += dt * un_acc;

    }
    Qj = Q_temp;
    Pj = P_temp;
    Vj = V_temp;


    {
      Eigen::Vector3d Pj_inte = Qi * imu_pre_integ.delta_p_ + Pi + Vi * sum_dt - 0.5 * G * sum_dt * sum_dt;
      Eigen::Quaterniond Qj_inte = (Qi * imu_pre_integ.delta_q_);
      Eigen::Vector3d Vj_inte = Qi * imu_pre_integ.delta_v_ + Vi - G * sum_dt;
      std::cout << "Pj pre inte: " << Pj_inte.transpose() << "; Pj: " << Pj.transpose() << std::endl;
      std::cout << "Qj pre inte: " << Qj_inte.coeffs().transpose() << "; Qj: " << Qj.coeffs().transpose()<< std::endl;
      std::cout << "Vj pre inte: " << Vj_inte.transpose() << "; Vj: " << Vj.transpose() << std::endl;
      Qj = Qj_inte;
      Pj = Pj_inte;
      Vj = Vj_inte;
    }


    // start_pose.P_ = start_pose.P_ + Eigen::Vector3d ::Random().normalized() * 1.3;
    // end_pose.P_ = end_pose.P_ + Eigen::Vector3d ::Random().normalized() * 1.3;

    Eigen::Vector3d dp_err = end_pose.P_ - start_pose.P_ ;
    // std::cout << "dp_err: " << dp_err.transpose() << "; dp_err normal: " << dp_err.norm() << std::endl;
    std::cout << "imu dp: " << imu_pre_integ.delta_p_.transpose() << ",normal: " << imu_pre_integ.delta_p_.norm() << ", gt dp: " <<
        dp_err.transpose() << ", dp_err: " <<dp_err.norm() << std::endl;
        /*
    Qi = start_pose.R_;
    Pi = start_pose.P_;
    Vi = start_pose.V_;
    Qj = end_pose.R_;
    Pj = end_pose.P_;
    Vj = end_pose.V_;*/
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
    speed_bias_i[3] = ba.x(), speed_bias_j[3] = baj.x();
    speed_bias_i[4] = ba.y(), speed_bias_j[4] = baj.y();
    speed_bias_i[5] = ba.z(), speed_bias_j[5] = baj.z();
    speed_bias_i[6] = bg.x(), speed_bias_j[6] = bgj.x();
    speed_bias_i[7] = bg.y(), speed_bias_j[7] = bgj.y();
    speed_bias_i[8] = bg.z(), speed_bias_j[8] = bgj.z();
    ceres::Problem problem;
    ceres::LocalParameterization *pose_i_local_param = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_pose_i, 7, pose_i_local_param);
    ceres::LocalParameterization *pose_j_local_param = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_pose_j, 7, pose_j_local_param);

    auto *imu_factor = new factor::IMUFactor(&imu_pre_integ);
    problem.AddResidualBlock(imu_factor, nullptr, param_pose_i, speed_bias_i, param_pose_j, speed_bias_j);
    // imu_factor->ComputeResidual({param_pose_i, speed_bias_i, param_pose_j, speed_bias_j});
    Eigen::Vector3d delta_p = end_pose.P_-start_pose.P_;
    // Qi * Qij = Qj; Qij = Qi.inverse() * Qj
    Eigen::Quaterniond delta_q = start_pose.R_.inverse() *end_pose.R_;
    ceres::CostFunction *global_pose_factor = factor::RelativeRTError::Create(delta_p.x(),
                                                                              delta_p.y(),
                                                                              delta_p.z(),
                                                                              delta_q.w(),
                                                                              delta_q.x(),
                                                                              delta_q.y(),
                                                                              delta_q.z(),
                                                                              0.1,
                                                                              0.01);
    // const T *const w_q_i, const T *ti, const T *w_q_j, const T *tj, T *residuals
    problem.AddResidualBlock(
        global_pose_factor, nullptr, param_pose_i , param_pose_j);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 7;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout.precision(10);

    Eigen::Vector3d before_P_err = end_pose.P_ - Pj;

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

    // std::cout << "P err: " << p_err.norm() << ", opt: " << p_err_opt.norm()
    //           << ", opt bg ba: " << p_err_opt_dba_dbg.norm() << "; " << p_err_opt.norm() - p_err_opt_dba_dbg.norm()
    //           << std::endl;
    Eigen::Vector3d opt_P_err = end_pose.P_ - Pj;

    std::cout << "before: " << before_P_err.transpose() << ", normal: " << before_P_err.norm() << "; opt :" <<
        opt_P_err.transpose() << opt_P_err.norm() << std::endl;

    result_traj.precision(20);
    /// t,x,y,z,x,y,z,w
    result_traj << start_time << " " ;
    result_traj.precision(8);
    result_traj <<  Pj.x() << " " <<
                    Pj.y() << " " <<
                    Pj.z() << " " <<
                    Qj.x() << " " <<
                    Qj.y() << " " <<
                    Qj.z() << " " <<
                    Qj.w() << std::endl;
    // std::cout << "pose_start_index: " << pose_start_index << ", pose_end_index: " << pose_end_index << std::endl;
    // std::cout << "Integration: " << imu_pre_integ.delta_p_.transpose() << std::endl;
    // std::cout << "Ctime: " << c_time << std::endl;
    // std::cout << "start time: " << start_time << ", end time: " << end_time << ", dt: " << end_time - start_time
    //           << ", FREQUENCY: " << 1. / (end_time - start_time) << std::endl;
  }
  std::cout << "Circle time: " << circle_time << std::endl;
}