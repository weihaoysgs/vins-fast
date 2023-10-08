//
// Created by weihao on 23-10-7.
//

#include "factor/imu/imu_integration_base.hpp"

namespace factor {

IntegrationBase::IntegrationBase(const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,
                                 const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)

  : acc_0_{acc_0}
  , gyr_0_{gyr_0}
  , linearized_acc_{acc_0}
  , linearized_gyr_{gyr_0}
  , linearized_ba_{linearized_ba}
  , linearized_bg_{linearized_bg}
{
  ReadNoiseParameter();
  noise_.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
  noise_.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
}

void IntegrationBase::ReadNoiseParameter()
{
  ACC_N = common::Setting::getSingleton()->Get<double>("acc_n");
  ACC_W = common::Setting::getSingleton()->Get<double>("acc_w");
  GYR_N = common::Setting::getSingleton()->Get<double>("gyr_n");
  GYR_W = common::Setting::getSingleton()->Get<double>("gyr_w");
  G.z() = common::Setting::getSingleton()->Get<double>("g_norm");
  // LOG(INFO) << "ACC_N: " << ACC_N << " ACC_W: " << ACC_W << " GYR_N: " << GYR_N << " GYR_W: " << GYR_W;
}

/// @brief Mid point integration
/// @param _dt delta time
/// @param _acc_0 last time acc
/// @param _gyr_0 last time gyroscope
/// @param _acc_1 current time acc
/// @param _gyr_1 current time gyroscope
/// @param delta_p delta translation
/// @param delta_q delta rotation
/// @param delta_v delta velocity
/// @param linearized_ba acc bias
/// @param linearized_bg gyroscope bias
/// @param result_delta_p updated delta translation
/// @param result_delta_q updated delta rotation
/// @param result_delta_v updated delta velocity
/// @param result_linearized_ba updated acc bias
/// @param result_linearized_bg updated gyroscope bias
/// @param update_jacobian update jacobian
void IntegrationBase::MidPointIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                          const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                          const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
                                          const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
                                          const Eigen::Vector3d &linearized_bg, Eigen::Vector3d &result_delta_p,
                                          Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                                          Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg,
                                          bool update_jacobian)
{
  Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
  Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
  result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
  Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
  result_delta_v = delta_v + un_acc * _dt;
  result_linearized_ba = linearized_ba;
  result_linearized_bg = linearized_bg;

  if (update_jacobian)
  {
    Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
    Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
    Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

    R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
    R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1), a_0_x(0), 0;
    R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1), a_1_x(0), 0;

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3) =
        -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
        -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
    F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
    F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
    F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(6, 3) =
        -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
        -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
    F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
    F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
    V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
    V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
    V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
    V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
    V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * _dt;
    V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * _dt;

    // step_jacobian_ = F;
    jacobian_ = F * jacobian_;
    covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
  }
}

void IntegrationBase::push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
  dt_buf_.push_back(dt);
  acc_buf_.push_back(acc);
  gyr_buf_.push_back(gyr);
  Propagate(dt, acc, gyr);
}

void IntegrationBase::Propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
{
  dt_ = _dt;
  acc_1_ = _acc_1;
  gyr_1_ = _gyr_1;
  Eigen::Vector3d result_delta_p;
  Eigen::Quaterniond result_delta_q;
  Eigen::Vector3d result_delta_v;
  Eigen::Vector3d result_linearized_ba;
  Eigen::Vector3d result_linearized_bg;
  MidPointIntegration(_dt,
                      acc_0_,
                      gyr_0_,
                      _acc_1,
                      _gyr_1,
                      delta_p_,
                      delta_q_,
                      delta_v_,
                      linearized_ba_,
                      linearized_bg_,
                      result_delta_p,
                      result_delta_q,
                      result_delta_v,
                      result_linearized_ba,
                      result_linearized_bg,
                      true);
  // CheckJacobian(_dt, acc_0_, gyr_0_, acc_1_, gyr_1_, delta_p_, delta_q_, delta_v_, linearized_ba_, linearized_bg_);
  delta_p_ = result_delta_p;
  delta_q_ = result_delta_q;
  delta_v_ = result_delta_v;
  linearized_ba_ = result_linearized_ba;
  linearized_bg_ = result_linearized_bg;
  delta_q_.normalize();
  sum_dt_ += dt_;
  acc_0_ = acc_1_;
  gyr_0_ = gyr_1_;
}

void IntegrationBase::CheckJacobian(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                    const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                    const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
                                    const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
                                    const Eigen::Vector3d &linearized_bg)
{
  Eigen::Vector3d result_delta_p;
  Eigen::Quaterniond result_delta_q;
  Eigen::Vector3d result_delta_v;
  Eigen::Vector3d result_linearized_ba;
  Eigen::Vector3d result_linearized_bg;
  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p,
                      delta_q,
                      delta_v,
                      linearized_ba,
                      linearized_bg,
                      result_delta_p,
                      result_delta_q,
                      result_delta_v,
                      result_linearized_ba,
                      result_linearized_bg,
                      0);

  Eigen::Vector3d turb_delta_p;
  Eigen::Quaterniond turb_delta_q;
  Eigen::Vector3d turb_delta_v;
  Eigen::Vector3d turb_linearized_ba;
  Eigen::Vector3d turb_linearized_bg;

  Eigen::Vector3d turb(0.0001, -0.003, 0.003);

  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p + turb,
                      delta_q,
                      delta_v,
                      linearized_ba,
                      linearized_bg,
                      turb_delta_p,
                      turb_delta_q,
                      turb_delta_v,
                      turb_linearized_ba,
                      turb_linearized_bg,
                      0);
  std::cout << "turb p       " << std::endl;
  std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
  std::cout << "p jacob diff " << (step_jacobian_.block<3, 3>(0, 0) * turb).transpose() << std::endl;
  std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
  std::cout << "q jacob diff " << (step_jacobian_.block<3, 3>(3, 0) * turb).transpose() << std::endl;
  std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
  std::cout << "v jacob diff " << (step_jacobian_.block<3, 3>(6, 0) * turb).transpose() << std::endl;
  std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
  std::cout << "ba jacob diff" << (step_jacobian_.block<3, 3>(9, 0) * turb).transpose() << std::endl;
  std::cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
  std::cout << "bg jacob diff " << (step_jacobian_.block<3, 3>(12, 0) * turb).transpose() << std::endl;

  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p,
                      delta_q * Eigen::Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2),
                      delta_v,
                      linearized_ba,
                      linearized_bg,
                      turb_delta_p,
                      turb_delta_q,
                      turb_delta_v,
                      turb_linearized_ba,
                      turb_linearized_bg,
                      0);
  std::cout << "turb q       " << std::endl;
  std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
  std::cout << "p jacob diff " << (step_jacobian_.block<3, 3>(0, 3) * turb).transpose() << std::endl;
  std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
  std::cout << "q jacob diff " << (step_jacobian_.block<3, 3>(3, 3) * turb).transpose() << std::endl;
  std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
  std::cout << "v jacob diff " << (step_jacobian_.block<3, 3>(6, 3) * turb).transpose() << std::endl;
  std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
  std::cout << "ba jacob diff" << (step_jacobian_.block<3, 3>(9, 3) * turb).transpose() << std::endl;
  std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
  std::cout << "bg jacob diff" << (step_jacobian_.block<3, 3>(12, 3) * turb).transpose() << std::endl;

  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p,
                      delta_q,
                      delta_v + turb,
                      linearized_ba,
                      linearized_bg,
                      turb_delta_p,
                      turb_delta_q,
                      turb_delta_v,
                      turb_linearized_ba,
                      turb_linearized_bg,
                      0);
  std::cout << "turb v       " << std::endl;
  std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
  std::cout << "p jacob diff " << (step_jacobian_.block<3, 3>(0, 6) * turb).transpose() << std::endl;
  std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
  std::cout << "q jacob diff " << (step_jacobian_.block<3, 3>(3, 6) * turb).transpose() << std::endl;
  std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
  std::cout << "v jacob diff " << (step_jacobian_.block<3, 3>(6, 6) * turb).transpose() << std::endl;
  std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
  std::cout << "ba jacob diff" << (step_jacobian_.block<3, 3>(9, 6) * turb).transpose() << std::endl;
  std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
  std::cout << "bg jacob diff" << (step_jacobian_.block<3, 3>(12, 6) * turb).transpose() << std::endl;

  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p,
                      delta_q,
                      delta_v,
                      linearized_ba + turb,
                      linearized_bg,
                      turb_delta_p,
                      turb_delta_q,
                      turb_delta_v,
                      turb_linearized_ba,
                      turb_linearized_bg,
                      0);
  std::cout << "turb ba       " << std::endl;
  std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
  std::cout << "p jacob diff " << (step_jacobian_.block<3, 3>(0, 9) * turb).transpose() << std::endl;
  std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
  std::cout << "q jacob diff " << (step_jacobian_.block<3, 3>(3, 9) * turb).transpose() << std::endl;
  std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
  std::cout << "v jacob diff " << (step_jacobian_.block<3, 3>(6, 9) * turb).transpose() << std::endl;
  std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
  std::cout << "ba jacob diff" << (step_jacobian_.block<3, 3>(9, 9) * turb).transpose() << std::endl;
  std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
  std::cout << "bg jacob diff" << (step_jacobian_.block<3, 3>(12, 9) * turb).transpose() << std::endl;

  MidPointIntegration(_dt,
                      _acc_0,
                      _gyr_0,
                      _acc_1,
                      _gyr_1,
                      delta_p,
                      delta_q,
                      delta_v,
                      linearized_ba,
                      linearized_bg + turb,
                      turb_delta_p,
                      turb_delta_q,
                      turb_delta_v,
                      turb_linearized_ba,
                      turb_linearized_bg,
                      0);
  std::cout << "turb bg       " << std::endl;
  std::cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << std::endl;
  std::cout << "p jacob diff " << (step_jacobian_.block<3, 3>(0, 12) * turb).transpose() << std::endl;
  std::cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << std::endl;
  std::cout << "q jacob diff " << (step_jacobian_.block<3, 3>(3, 12) * turb).transpose() << std::endl;
  std::cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << std::endl;
  std::cout << "v jacob diff " << (step_jacobian_.block<3, 3>(6, 12) * turb).transpose() << std::endl;
  std::cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << std::endl;
  std::cout << "ba jacob diff" << (step_jacobian_.block<3, 3>(9, 12) * turb).transpose() << std::endl;
  std::cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << std::endl;
  std::cout << "bg jacob diff" << (step_jacobian_.block<3, 3>(12, 12) * turb).transpose() << std::endl;
}

Eigen::Matrix<double, 15, 1> IntegrationBase::Evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
                                                       const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
                                                       const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
                                                       const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
                                                       const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
{
  Eigen::Matrix<double, 15, 1> residuals;

  Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(O_P, O_BA);
  Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(O_P, O_BG);

  Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(O_R, O_BG);

  Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(O_V, O_BA);
  Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(O_V, O_BG);

  Eigen::Vector3d dba = Bai - linearized_ba_;
  Eigen::Vector3d dbg = Bgi - linearized_bg_;

  Eigen::Quaterniond corrected_delta_q = delta_q_ * common::Algorithm::DeltaQ(dq_dbg * dbg);
  Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
  Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

  residuals.block<3, 1>(O_P, 0) =
      Qi.inverse() * (0.5 * G * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
  residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
  residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt_ + Vj - Vi) - corrected_delta_v;
  residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
  residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
  return residuals;
}

} // namespace factor
