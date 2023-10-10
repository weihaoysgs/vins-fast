#include "estimator/estimator.hpp"

namespace estimator {

Estimator::Estimator()
{
  /// first call to clear member variable value
  ClearState();
  SetParameter();

  visualization_ = std::make_shared<common::Visualization>();

  feature_tracker_ = std::make_shared<FeatureTracker>();
  feature_tracker_->ReadIntrinsicParameter(camera_calib_file_path_);

  feature_manager_ = std::make_shared<FeatureManager>();
  feature_manager_->SetRic(ric_);

  std::thread t_frontend_process = std::thread(&Estimator::tFrontendProcess, this);
  t_frontend_process.detach();
  std::thread t_backend_process = std::thread(&Estimator::tBackendProcess, this);
  t_backend_process.detach();
}

void Estimator::ClearState()
{
  frame_count_ = 0;
  while (!feature_buf_.empty())
    feature_buf_.pop();
  previous_time_ = -1;
  current_time_ = 0;
  input_image_cnt_ = 0;
  time_diff_ = 0;
  solver_flag_ = SolverFlag::INITIAL;
  for (int i = 0; i < 2; i++)
  {
    tic_[i] = Eigen::Vector3d::Zero();
    ric_[i] = Eigen::Matrix3d::Identity();
  }
  for (int i = 0; i < WINDOW_SIZE + 1; i++)
  {
    Rs_[i].setIdentity();
    Ps_[i].setZero();
    Vs_[i].setZero();
    Bas_[i].setZero();
    Bgs_[i].setZero();
  }
}

void Estimator::SetParameter()
{
  g_.z() = common::Setting::getSingleton()->Get<double>("g_norm");
  USE_IMU = common::Setting::getSingleton()->Get<int>("imu");
  IMG0_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image0_topic");
  IMG1_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image1_topic");
  IMU_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("imu_topic");
  LOG(INFO) << "IMG1 TOPIC: " << IMG1_TOPIC_NAME << ", IMG0 TOPIC: " << IMG1_TOPIC_NAME
            << ", IMU_TOPIC_NAME: " << IMU_TOPIC_NAME << ", USE_IMU:" << USE_IMU << ", WINDOW_SIZE:" << WINDOW_SIZE;

  ReadImuCameraExternalParam();
  factor::ProjectionTwoFrameOneCamFactor::sqrt_info_ = 460.0 / 1.5 * Eigen::Matrix2d::Identity();
  factor::ProjectionTwoFrameTwoCamFactor::sqrt_info_ = 460.0 / 1.5 * Eigen::Matrix2d::Identity();
  factor::ProjectionOneFrameTwoCamFactor::sqrt_info_ = 460.0 / 1.2 * Eigen::Matrix2d::Identity();
}

void Estimator::ReadImuCameraExternalParam()
{
  cv::Mat cv_T_0, cv_T_1;
  Eigen::Matrix4d T0, T1;
  common::Setting::getSingleton()->getFile()["body_T_cam0"] >> cv_T_0;
  cv::cv2eigen(cv_T_0, T0);
  ric_[0] = T0.block<3, 3>(0, 0);
  tic_[0] = T0.block<3, 1>(0, 3);
  common::Setting::getSingleton()->getFile()["body_T_cam1"] >> cv_T_1;
  cv::cv2eigen(cv_T_1, T1);
  ric_[1] = T1.block<3, 3>(0, 0);
  tic_[1] = T1.block<3, 1>(0, 3);
  LOG(INFO) << "Cam0 R:\n" << ric_[0] << "\n t: " << tic_[0].transpose();
  LOG(INFO) << "Cam1 R:\n" << ric_[1] << "\n t: " << tic_[1].transpose();

  std::string cam0_calib_file_path = common::Setting::getSingleton()->Get<std::string>("cam0_calib");
  std::string cam1_calib_file_path = common::Setting::getSingleton()->Get<std::string>("cam1_calib");
  assert(common::FileExists(cam0_calib_file_path) && common::FileExists(cam1_calib_file_path));
  camera_calib_file_path_.emplace_back(cam0_calib_file_path);
  camera_calib_file_path_.emplace_back(cam1_calib_file_path);
  LOG(INFO) << "Cam0 calib file: " << cam0_calib_file_path;
  LOG(INFO) << "Cam1 calib file: " << cam1_calib_file_path;
}

void Estimator::tFrontendProcess()
{
  while (true)
  {
    cv::Mat image0, image1;
    double time = 0;
    img_buf_mutex_.lock();
    if (!img0_buf_.empty() && !img1_buf_.empty())
    {
      double time0 = img0_buf_.front()->header.stamp.toSec();
      double time1 = img1_buf_.front()->header.stamp.toSec();
      // 0.003s sync tolerance
      if (time0 < time1 - 0.003)
      {
        img0_buf_.pop();
        LOG(ERROR) << "Throw img0";
      }
      else if (time0 > time1 + 0.003)
      {
        img1_buf_.pop();
        LOG(ERROR) << "Throw img1";
      }
      else
      {
        time = img0_buf_.front()->header.stamp.toSec();
        image0 = common::GetImageFromROSMsg(img0_buf_.front());
        img0_buf_.pop();
        image1 = common::GetImageFromROSMsg(img1_buf_.front());
        img1_buf_.pop();
      }
    }
    img_buf_mutex_.unlock();
    if (!image0.empty() && !image1.empty())
    {
      FrontendTracker(time, image0, image1);
    }
  }
}

void Estimator::FrontendTracker(double t, const cv::Mat &img0, const cv::Mat &img1)
{
  input_image_cnt_++;
  {
    std::unique_lock<std::mutex> lck(current_img_mutex_);
    current_img_ = img0;
  }

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> frame_feature;
  frame_feature = feature_tracker_->TrackImage(t, img0, img1);
  if (input_image_cnt_ % 2 == 0)
  {
    std::unique_lock<std::mutex> lck(feature_buf_mutex_);
    feature_buf_.push(std::make_pair(t, frame_feature));
  }
}

bool Estimator::IMUAvailable(double t)
{
  if (!acc_buf_.empty() && t <= acc_buf_.back().first)
    return true;
  else
    return false;
}

void Estimator::tBackendProcess()
{
  while (true)
  {
    std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>> feature_frame;
    std::vector<std::pair<double, Eigen::Vector3d>> acc_vector, gyr_vector;
    if (!feature_buf_.empty())
    {
      feature_buf_mutex_.lock();
      feature_frame = feature_buf_.front();
      feature_buf_.pop();
      feature_buf_mutex_.unlock();
      current_time_ = feature_frame.first + time_diff_;
      ///////////////////////////
      /// Get IMU data
      while (true)
      {
        if (!USE_IMU || IMUAvailable(current_time_))
        {
          break;
        }
        else
        {
          LOG(WARNING) << "wait for imu ... ";
          std::chrono::milliseconds dura(1);
          std::this_thread::sleep_for(dura);
        }
      }
      if (USE_IMU)
      {
        GetIMUInterval(previous_time_, current_time_, acc_vector, gyr_vector);
        if (!init_first_pose_flag_)
        {
          InitFirstIMUPose(acc_vector);
        }
        for (size_t i = 0; i < acc_vector.size(); i++)
        {
          double dt;
          if (i == 0)
            dt = acc_vector[i].first - previous_time_;
          else if (i == acc_vector.size() - 1)
            dt = current_time_ - acc_vector[i - 1].first;
          else
            dt = acc_vector[i].first - acc_vector[i - 1].first;
          ProcessIMU(acc_vector[i].first, dt, acc_vector[i].second, gyr_vector[i].second);
        }
      }

      ///////////////////////////
      ProcessImage(feature_frame.second, current_time_);

      previous_time_ = current_time_;

      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time(feature_frame.first);

      visualization_->PublishPath(header.stamp.toSec(), Ps_, Rs_, WINDOW_SIZE);
      visualization_->PublishTrackImage(feature_tracker_->getDrawTrackResultImg(), current_time_);
    }

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

void Estimator::ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                             const double time)
{
  if (feature_manager_->AddFeatureAndCheckParallax(frame_count_, image, time_diff_))
    marg_flag_ = MarginalizationFlag::MARGIN_OLD;
  else
    marg_flag_ = MarginalizationFlag::MARGIN_SECOND_NEW;

  if (solver_flag_ == SolverFlag::INITIAL)
  {
    if (USE_IMU)
    {
      feature_manager_->InitFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
      feature_manager_->TriangulatePts(Ps_, Rs_, tic_, ric_);
      Optimization();

      if (frame_count_ == WINDOW_SIZE)
      {
        solver_flag_ = NON_LINEAR;
        SlideWindow();
      }
    }

    if (!USE_IMU)
    {
      feature_manager_->InitFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
      feature_manager_->TriangulatePts(Ps_, Rs_, tic_, ric_);
      Optimization();

      if (frame_count_ == WINDOW_SIZE)
      {
        solver_flag_ = NON_LINEAR;
        SlideWindow();
      }
    }

    if (frame_count_ < WINDOW_SIZE)
    {
      frame_count_++;
      int previous_frame = frame_count_ - 1;
      Ps_[frame_count_] = Ps_[previous_frame];
      Vs_[frame_count_] = Vs_[previous_frame];
      Rs_[frame_count_] = Rs_[previous_frame];
      Bas_[frame_count_] = Bas_[previous_frame];
      Bgs_[frame_count_] = Bgs_[previous_frame];
    }
  }

  else
  {
    if (!USE_IMU)
      feature_manager_->InitFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
    feature_manager_->TriangulatePts(Ps_, Rs_, tic_, ric_);
    common::TicToc tim;
    Optimization();
    // LOG(INFO) << "Optimization cost " << tim.tEnd() << " ms";
    std::set<int> remove_ids;
    feature_manager_->OutliersRejection(remove_ids, Rs_, Ps_, ric_, tic_);
    feature_manager_->RemoveOutlier(remove_ids);

    SlideWindow();
  }
}
void Estimator::SlideWindow()
{
  if (marg_flag_ == MARGIN_OLD)
  {
    back_R0_ = Rs_[0];
    back_P0_ = Ps_[0];
    if (frame_count_ == WINDOW_SIZE)
    {
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        headers_[i] = headers_[i + 1];
        Rs_[i].swap(Rs_[i + 1]);
        Ps_[i].swap(Ps_[i + 1]);
        if (USE_IMU)
        {
          std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

          dt_buf_[i].swap(dt_buf_[i + 1]);
          linear_acceleration_buf_[i].swap(linear_acceleration_buf_[i + 1]);
          angular_velocity_buf_[i].swap(angular_velocity_buf_[i + 1]);

          Vs_[i].swap(Vs_[i + 1]);
          Bas_[i].swap(Bas_[i + 1]);
          Bgs_[i].swap(Bgs_[i + 1]);
        }
      }
      headers_[WINDOW_SIZE] = headers_[WINDOW_SIZE - 1];
      Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
      Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];
      if (USE_IMU)
      {
        Vs_[WINDOW_SIZE] = Vs_[WINDOW_SIZE - 1];
        Bas_[WINDOW_SIZE] = Bas_[WINDOW_SIZE - 1];
        Bgs_[WINDOW_SIZE] = Bgs_[WINDOW_SIZE - 1];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] =
            new factor::IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();
      }

      SlideWindowOld();
    }
  }
  else if (marg_flag_ == MARGIN_SECOND_NEW)
  {
    if (frame_count_ == WINDOW_SIZE)
    {
      headers_[frame_count_ - 1] = headers_[frame_count_];
      Ps_[frame_count_ - 1] = Ps_[frame_count_];
      Rs_[frame_count_ - 1] = Rs_[frame_count_];

      if (USE_IMU)
      {
        for (unsigned int i = 0; i < dt_buf_[frame_count_].size(); i++)
        {
          double tmp_dt = dt_buf_[frame_count_][i];
          Eigen::Vector3d tmp_linear_acceleration = linear_acceleration_buf_[frame_count_][i];
          Eigen::Vector3d tmp_angular_velocity = angular_velocity_buf_[frame_count_][i];

          pre_integrations_[frame_count_ - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

          dt_buf_[frame_count_ - 1].push_back(tmp_dt);
          linear_acceleration_buf_[frame_count_ - 1].push_back(tmp_linear_acceleration);
          angular_velocity_buf_[frame_count_ - 1].push_back(tmp_angular_velocity);
        }

        Vs_[frame_count_ - 1] = Vs_[frame_count_];
        Bas_[frame_count_ - 1] = Bas_[frame_count_];
        Bgs_[frame_count_ - 1] = Bgs_[frame_count_];

        delete pre_integrations_[WINDOW_SIZE];
        pre_integrations_[WINDOW_SIZE] =
            new factor::IntegrationBase{acc_0_, gyr_0_, Bas_[WINDOW_SIZE], Bgs_[WINDOW_SIZE]};

        dt_buf_[WINDOW_SIZE].clear();
        linear_acceleration_buf_[WINDOW_SIZE].clear();
        angular_velocity_buf_[WINDOW_SIZE].clear();
      }

      SlideWindowNew();
    }
  }
}

void Estimator::SlideWindowOld()
{
  if (solver_flag_ == NON_LINEAR)
  {
    Eigen::Matrix3d R0, R1;
    Eigen::Vector3d P0, P1;
    R0 = back_R0_ * ric_[0];
    R1 = Rs_[0] * ric_[0];
    P0 = back_P0_ + back_R0_ * tic_[0];
    P1 = Ps_[0] + Rs_[0] * tic_[0];
    feature_manager_->RemoveBackShiftDepth(R0, P0, R1, P1);
  }
  else
  {
    feature_manager_->RemoveBack();
  }
}

void Estimator::SlideWindowNew()
{
  feature_manager_->RemoveFront(frame_count_, WINDOW_SIZE);
}

void Estimator::Optimization()
{
  vector2double();
  ceres::Problem problem;
  ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
  for (int i = 0; i < frame_count_ + 1; i++)
  {
    ceres::LocalParameterization *local_pose_parameter = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_pose_[i], SIZE_POSE, local_pose_parameter);
    if (USE_IMU)
    {
      problem.AddParameterBlock(param_speed_bias_[i], SIZE_SPEED_BIAS);
    }
  }
  if (!USE_IMU)
  {
    problem.SetParameterBlockConstant(param_pose_[0]);
  }

  for (int i = 0; i < 2; i++)
  {
    ceres::LocalParameterization *local_cam_pose_parameter = new factor::PoseLocalParameterization();
    problem.AddParameterBlock(param_ex_pose_[i], SIZE_POSE, local_cam_pose_parameter);
    {
      problem.SetParameterBlockConstant(param_ex_pose_[i]);
    }
  }

  problem.AddParameterBlock(param_td_[0], 1);
  {
    problem.SetParameterBlockConstant(param_td_[0]);
  }

  if (last_marginalization_info_ && last_marginalization_info_->valid_)
  {
    // construct new marginalization_factor
    auto *marginalization_factor = new factor::MarginalizationFactor(last_marginalization_info_);
    problem.AddResidualBlock(marginalization_factor, nullptr, last_marginalization_parameter_blocks_);
  }
  if (USE_IMU)
  {
    for (int i = 0; i < frame_count_; i++)
    {
      int j = i + 1;
      if (pre_integrations_[j]->sum_dt_ > 10.0)
        continue;
      auto *imu_factor = new factor::IMUFactor(pre_integrations_[j]);
      problem.AddResidualBlock(
          imu_factor, nullptr, param_pose_[i], param_speed_bias_[i], param_pose_[j], param_speed_bias_[j]);
    }
  }
  int feature_index = -1;
  for (auto &id_observed_feature : feature_manager_->getFeatures())
  {
    id_observed_feature.used_num_ = id_observed_feature.feature_per_frame_.size();
    if (id_observed_feature.used_num_ < 4)
      continue;
    ++feature_index;
    int window_i = id_observed_feature.start_frame_, window_j = window_i - 1;

    Eigen::Vector3d pts_i = id_observed_feature.feature_per_frame_[0].point_;

    for (auto &id_per_frame : id_observed_feature.feature_per_frame_)
    {
      window_j++;
      if (window_i != window_j)
      {
        // clang-format off
        Eigen::Vector3d pts_j = id_per_frame.point_;
        auto *residual_block = new factor::ProjectionTwoFrameOneCamFactor(pts_i, pts_j,
            id_observed_feature.feature_per_frame_[0].velocity_, id_per_frame.velocity_,
                  id_observed_feature.feature_per_frame_[0].cur_td_, id_per_frame.cur_td_);
        problem.AddResidualBlock(residual_block, loss_function, param_pose_[window_i],
                                  param_pose_[window_j], param_ex_pose_[0],
                                 param_feature_[feature_index], param_td_[0]);

      }
      if (id_per_frame.track_right_success_)
      {
        Eigen::Vector3d pts_j_right = id_per_frame.point_right_;
        if (window_i != window_j)
        {
          auto *residual = new factor::ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right,
                           id_observed_feature.feature_per_frame_[0].velocity_, id_per_frame.velocity_right_,
                                     id_observed_feature.feature_per_frame_[0].cur_td_, id_per_frame.cur_td_);

          problem.AddResidualBlock(residual, loss_function, param_pose_[window_i], param_pose_[window_j],
                                   param_ex_pose_[0], param_ex_pose_[1], param_feature_[feature_index], param_td_[0]);
        }
        else
        {
          auto *residual = new factor::ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right,
                                id_observed_feature.feature_per_frame_[0].velocity_,
                                 id_per_frame.velocity_, id_observed_feature.feature_per_frame_[0].cur_td_,
                                     id_per_frame.cur_td_);
          problem.AddResidualBlock(residual, loss_function, param_ex_pose_[0], param_ex_pose_[1],
                                   param_feature_[feature_index], param_td_[0]);
          // clang-format on
        }
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 8;
  options.max_solver_time_in_seconds = 0.04;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  double2vector();

  common::TicToc tim;
  PrepareMarginalizationFactor();
  // std::string marg_flag = marg_flag_ == MARGIN_OLD ? "MARGIN_OLD" : "MARGIN_SECOND_NEW";
  // LOG(INFO) << marg_flag + " marginalization cost " << tim.tEnd() << " ms";
}

void Estimator::PrepareMarginalizationFactor()
{
  ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
  if (frame_count_ < WINDOW_SIZE)
    return;
  if (marg_flag_ == MARGIN_OLD)
  {
    auto *marginalization_info = new factor::MarginalizationInfo();
    vector2double();
    if (last_marginalization_info_ && last_marginalization_info_->valid_)
    {
      std::vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++)
      {
        if (last_marginalization_parameter_blocks_[i] == param_pose_[0] ||
            last_marginalization_parameter_blocks_[i] == param_speed_bias_[0])
          drop_set.push_back(i);
      }
      auto *marginalization_factor = new factor::MarginalizationFactor(last_marginalization_info_);
      auto *residual_block_info = new factor::MargResidualBlockInfo(
          marginalization_factor, nullptr, last_marginalization_parameter_blocks_, drop_set);
      residual_block_info->residual_block_name_ = "marg_factor";
      marginalization_info->AddResidualBlockInfo(residual_block_info);
    }

    if (USE_IMU)
    {
      if (pre_integrations_[1]->sum_dt_ < 10.0)
      {
        auto *imu_factor = new factor::IMUFactor(pre_integrations_[1]);
        auto *residual_block_info = new factor::MargResidualBlockInfo(
            imu_factor,
            nullptr,
            std::vector<double *>{param_pose_[0], param_speed_bias_[0], param_pose_[1], param_speed_bias_[1]},
            std::vector<int>{0, 1});
        marginalization_info->AddResidualBlockInfo(residual_block_info);
      }
    }

    int feature_index = -1;
    for (auto &it_per_id : feature_manager_->getFeatures())
    {
      it_per_id.used_num_ = static_cast<int>(it_per_id.feature_per_frame_.size());
      if (it_per_id.used_num_ < 4)
        continue;
      ++feature_index;
      int window_i = it_per_id.start_frame_, window_j = window_i - 1;

      if (window_i != 0)
        continue;

      Eigen::Vector3d pts_i = it_per_id.feature_per_frame_[0].point_;
      for (auto &it_per_frame : it_per_id.feature_per_frame_)
      {
        window_j++;
        if (window_i != window_j)
        {
          // clang-format off
          Eigen::Vector3d pts_j = it_per_frame.point_;
          auto *f_td = new factor::ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame_[0].velocity_,
                            it_per_frame.velocity_, it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
          auto *residual_block_info = new factor::MargResidualBlockInfo(f_td, loss_function,
                                      std::vector<double *>{param_pose_[0], param_pose_[window_j], param_ex_pose_[0],
                                       param_feature_[feature_index], param_td_[0]}, std::vector<int>{0, 3});
          marginalization_info->AddResidualBlockInfo(residual_block_info);
        }
        if(it_per_frame.track_right_success_)
        {
          Eigen::Vector3d pts_j_right = it_per_frame.point_right_;
          if(window_i != window_j)
          {
            auto *f = new factor::ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right,
                           it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocity_right_,
                                     it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
            auto *residual_block_info = new factor::MargResidualBlockInfo(f, loss_function,
                                  std::vector<double *>{param_pose_[0], param_pose_[window_j], param_ex_pose_[0],
                                                  param_ex_pose_[1], param_feature_[feature_index], param_td_[0]}, std::vector<int>{0, 4});
            marginalization_info->AddResidualBlockInfo(residual_block_info);
          }
          else
          {
            auto *f = new factor::ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right,
                        it_per_id.feature_per_frame_[0].velocity_, it_per_frame.velocity_right_,
                          it_per_id.feature_per_frame_[0].cur_td_, it_per_frame.cur_td_);
            auto *residual_block_info = new factor::MargResidualBlockInfo(f, loss_function,
                         std::vector<double *>{param_ex_pose_[0],
                             param_ex_pose_[1], param_feature_[feature_index], param_td_[0]}, std:: vector<int>{2});
            marginalization_info->AddResidualBlockInfo(residual_block_info);
            // clang-format on
          }
        }
      }
    }
    // auto marg_factor_num = std::count_if(marginalization_info->factors_.begin(),
    //            marginalization_info->factors_.end(),[](const factor::MargResidualBlockInfo *residual){
    //   return residual->residual_block_name_ == "marg_factor";
    // });
    marginalization_info->PreMarginalize();
    marginalization_info->Marginalize();
    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
      addr_shift[reinterpret_cast<long>(param_pose_[i])] = param_pose_[i - 1];
      if (USE_IMU)
      {
        addr_shift[reinterpret_cast<long>(param_speed_bias_[i])] = param_speed_bias_[i - 1];
      }
    }
    for (int i = 0; i < 2; i++)
      addr_shift[reinterpret_cast<long>(param_ex_pose_[i])] = param_ex_pose_[i];

    addr_shift[reinterpret_cast<long>(param_td_[0])] = param_td_[0];

    std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info_)
      delete last_marginalization_info_;
    last_marginalization_info_ = marginalization_info;
    last_marginalization_parameter_blocks_ = parameter_blocks;
  }
  else
  {
    if (last_marginalization_info_ && std::count(std::begin(last_marginalization_parameter_blocks_),
                                                 std::end(last_marginalization_parameter_blocks_),
                                                 param_pose_[WINDOW_SIZE - 1]))
    {
      auto *marginalization_info = new factor::MarginalizationInfo();
      vector2double();
      if (last_marginalization_info_ && last_marginalization_info_->valid_)
      {
        std::vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++)
        {
          // ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks_[i] == param_pose_[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        auto *marginalization_factor = new factor::MarginalizationFactor(last_marginalization_info_);
        auto *residual_block_info = new factor::MargResidualBlockInfo(
            marginalization_factor, nullptr, last_marginalization_parameter_blocks_, drop_set);

        marginalization_info->AddResidualBlockInfo(residual_block_info);
      }
      marginalization_info->PreMarginalize();
      marginalization_info->Marginalize();

      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++)
      {
        if (i == WINDOW_SIZE - 1)
          continue;
        else if (i == WINDOW_SIZE)
        {
          addr_shift[reinterpret_cast<long>(param_pose_[i])] = param_pose_[i - 1];
          if (USE_IMU)
          {
            addr_shift[reinterpret_cast<long>(param_speed_bias_[i])] = param_speed_bias_[i - 1];
          }
        }
        else
        {
          addr_shift[reinterpret_cast<long>(param_pose_[i])] = param_pose_[i];
          if (USE_IMU)
          {
            addr_shift[reinterpret_cast<long>(param_speed_bias_[i])] = param_speed_bias_[i];
          }
        }
      }
      for (int i = 0; i < 2; i++)
        addr_shift[reinterpret_cast<long>(param_ex_pose_[i])] = param_ex_pose_[i];

      addr_shift[reinterpret_cast<long>(param_td_[0])] = param_td_[0];

      std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info_)
        delete last_marginalization_info_;
      last_marginalization_info_ = marginalization_info;
      last_marginalization_parameter_blocks_ = parameter_blocks;
    }
  }
}

void Estimator::vector2double()
{
  for (int i = 0; i <= WINDOW_SIZE; i++)
  {
    param_pose_[i][0] = Ps_[i].x();
    param_pose_[i][1] = Ps_[i].y();
    param_pose_[i][2] = Ps_[i].z();
    Eigen::Quaterniond q{Rs_[i]};
    param_pose_[i][3] = q.x();
    param_pose_[i][4] = q.y();
    param_pose_[i][5] = q.z();
    param_pose_[i][6] = q.w();
    if (USE_IMU)
    {
      param_speed_bias_[i][0] = Vs_[i].x();
      param_speed_bias_[i][1] = Vs_[i].y();
      param_speed_bias_[i][2] = Vs_[i].z();

      param_speed_bias_[i][3] = Bas_[i].x();
      param_speed_bias_[i][4] = Bas_[i].y();
      param_speed_bias_[i][5] = Bas_[i].z();

      param_speed_bias_[i][6] = Bgs_[i].x();
      param_speed_bias_[i][7] = Bgs_[i].y();
      param_speed_bias_[i][8] = Bgs_[i].z();
    }
  }
  for (int i = 0; i < 2; i++)
  {
    param_ex_pose_[i][0] = tic_[i].x();
    param_ex_pose_[i][1] = tic_[i].y();
    param_ex_pose_[i][2] = tic_[i].z();
    Eigen::Quaterniond q{ric_[i]};
    param_ex_pose_[i][3] = q.x();
    param_ex_pose_[i][4] = q.y();
    param_ex_pose_[i][5] = q.z();
    param_ex_pose_[i][6] = q.w();
  }

  Eigen::VectorXd dep = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    param_feature_[i][0] = dep(i);

  param_td_[0][0] = time_diff_;
}

void Estimator::double2vector()
{
  /// Rotation and position before optimization
  // clang-format off
  Eigen::Vector3d origin_R0 = common::Algorithm::R2ypr(Rs_[0]);
  Eigen::Vector3d origin_P0 = Ps_[0];
  if (USE_IMU)
  {
    /// Optimized rotation
    Eigen::Vector3d origin_R00 = common::Algorithm::R2ypr(
        Eigen::Quaterniond(param_pose_[0][6], param_pose_[0][3], param_pose_[0][4], param_pose_[0][5]).toRotationMatrix());
    /// Change amount of yaw axis after optimization
    double y_diff = origin_R0.x() - origin_R00.x();
    Eigen::Matrix3d rot_diff = common::Algorithm::ypr2R(Eigen::Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
      LOG(WARNING) << "euler singular point!";
      rot_diff = Rs_[0] * Eigen::Quaterniond(param_pose_[0][6], param_pose_[0][3], param_pose_[0][4], param_pose_[0][5]).toRotationMatrix().transpose();
    }
    /// system param
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      Rs_[i] = rot_diff * Eigen::Quaterniond(param_pose_[i][6], param_pose_[i][3], param_pose_[i][4], param_pose_[i][5]).normalized().toRotationMatrix();
      Ps_[i] = rot_diff * Eigen::Vector3d(param_pose_[i][0] - param_pose_[0][0],
                                          param_pose_[i][1] - param_pose_[0][1],
                                          param_pose_[i][2] - param_pose_[0][2]) + origin_P0;
      Vs_[i] = rot_diff * Eigen::Vector3d(param_speed_bias_[i][0], param_speed_bias_[i][1], param_speed_bias_[i][2]);
      Bas_[i] = Eigen::Vector3d(param_speed_bias_[i][3], param_speed_bias_[i][4], param_speed_bias_[i][5]);
      Bgs_[i] = Eigen::Vector3d(param_speed_bias_[i][6], param_speed_bias_[i][7], param_speed_bias_[i][8]);
    }
    /// external param
    for (int i = 0; i < 2; i++)
    {
      // tic_[i] = Eigen::Vector3d(param_ex_pose_[i][0], param_ex_pose_[i][1], param_ex_pose_[i][2]);
      // ric_[i] = Eigen::Quaterniond(param_ex_pose_[i][6], param_ex_pose_[i][3], param_ex_pose_[i][4], param_ex_pose_[i][5]).normalized().toRotationMatrix();
    }
    /// time differ param
    time_diff_ = param_td_[0][0];
  }
  else
  {
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      Rs_[i] = Eigen::Quaterniond(param_pose_[i][6], param_pose_[i][3], param_pose_[i][4], param_pose_[i][5]).normalized().toRotationMatrix();
      Ps_[i] = Eigen::Vector3d(param_pose_[i][0], param_pose_[i][1], param_pose_[i][2]);
    }
  }

  Eigen::VectorXd dep = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    dep(i) = param_feature_[i][0];
  feature_manager_->setDepth(dep);
  // clang-format on
}

bool Estimator::GetIMUInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>> &acc_vector,
                               std::vector<std::pair<double, Eigen::Vector3d>> &gyr_vector)
{
  if (acc_buf_.empty())
  {
    LOG(WARNING) << "not receive imu";
    return false;
  }
  ///              t0        t1
  /// [front] * * * * * * * * * *  [back]
  if (t1 <= acc_buf_.back().first)
  {
    while (acc_buf_.front().first <= t0)
    {
      acc_buf_.pop();
      gyr_buf_.pop();
    }
    ///  t0        t1
    ///  * * * * * * * * *
    while (acc_buf_.front().first < t1)
    {
      acc_vector.push_back(acc_buf_.front());
      acc_buf_.pop();
      gyr_vector.push_back(gyr_buf_.front());
      gyr_buf_.pop();
    }
    acc_vector.push_back(acc_buf_.front());
    gyr_vector.push_back(gyr_buf_.front());
  }
  else
  {
    LOG(WARNING) << "wait for imu";
    return false;
  }
  return true;
}

void Estimator::IMUCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
  double t = imu_msg->header.stamp.toSec();
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Eigen::Vector3d acc(dx, dy, dz);
  Eigen::Vector3d gyr(rx, ry, rz);
  std::unique_lock<std::mutex> lck(imu_buf_mutex_);
  acc_buf_.push(std::make_pair(t, acc));
  gyr_buf_.push(std::make_pair(t, gyr));
}

void Estimator::InitFirstIMUPose(std::vector<std::pair<double, Eigen::Vector3d>> &acc_vector)
{
  LOG(INFO) << "init first imu pose";
  init_first_pose_flag_ = true;
  Eigen::Vector3d aver_ccc(0, 0, 0);
  for (auto &i : acc_vector)
  {
    aver_ccc = aver_ccc + i.second;
  }
  aver_ccc = aver_ccc / acc_vector.size();
  LOG(INFO) << "averge acc: " << aver_ccc.transpose();
  Eigen::Matrix3d R0 = common::Algorithm::g2R(aver_ccc);
  Rs_[0] = R0;
  LOG(INFO) << "init R0\n " << Rs_[0];
}

void Estimator::ProcessIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration,
                           const Eigen::Vector3d &angular_velocity)
{
  if (first_imu_)
  {
    first_imu_ = false;
    acc_0_ = linear_acceleration;
    gyr_0_ = angular_velocity;
  }

  if (!pre_integrations_[frame_count_])
  {
    pre_integrations_[frame_count_] =
        new factor::IntegrationBase{acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_]};
  }

  if (frame_count_ != 0)
  {
    pre_integrations_[frame_count_]->push_back(dt, linear_acceleration, angular_velocity);
    dt_buf_[frame_count_].push_back(dt);
    linear_acceleration_buf_[frame_count_].push_back(linear_acceleration);
    angular_velocity_buf_[frame_count_].push_back(angular_velocity);

    int j = frame_count_;
    Eigen::Vector3d un_acc_0 = Rs_[j] * (acc_0_ - Bas_[j]) - g_; /// a^' = R^T * (a-g)
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0_ + angular_velocity) - Bgs_[j];
    Rs_[j] *= common::Algorithm::DeltaQ(un_gyr * dt).toRotationMatrix();
    Eigen::Vector3d un_acc_1 = Rs_[j] * (linear_acceleration - Bas_[j]) - g_;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps_[j] += dt * Vs_[j] + 0.5 * dt * dt * un_acc;
    Vs_[j] += dt * un_acc;
  }
  acc_0_ = linear_acceleration;
  gyr_0_ = angular_velocity;
}

}; // namespace estimator
