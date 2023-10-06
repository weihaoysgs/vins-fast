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

void Estimator::tBackendProcess()
{
  while (true)
  {
    std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>> feature_frame;
    if (!feature_buf_.empty())
    {
      feature_buf_mutex_.lock();
      feature_frame = feature_buf_.front();
      feature_buf_.pop();
      feature_buf_mutex_.unlock();
      current_time_ = feature_frame.first + time_diff_;
      ///////////////////////////
      /// Get IMU data
      ///////////////////////////
      ProcessImage(feature_frame.second, current_time_);

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
    if (!USE_IMU)
    {
      feature_manager_->InitFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
      /// 得到了两帧的位姿了，那么就可以三角化同时被这两帧看到的特征点了,注意当前帧新提取的仍然不能被三角化
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
    feature_manager_->InitFramePoseByPnP(frame_count_, Ps_, Rs_, tic_, ric_);
    feature_manager_->TriangulatePts(Ps_, Rs_, tic_, ric_);

    Optimization();
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
      }
      headers_[WINDOW_SIZE] = headers_[WINDOW_SIZE - 1];
      Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
      Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];

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
  }

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
  /// 优化前的旋转和位置
  Eigen::Vector3d origin_R0 = common::Algorithm::R2ypr(Rs_[0]);
  Eigen::Vector3d origin_P0 = Ps_[0];
  {
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
      Rs_[i] = Eigen::Quaterniond(param_pose_[i][6], param_pose_[i][3], param_pose_[i][4], param_pose_[i][5])
                   .normalized()
                   .toRotationMatrix();
      Ps_[i] = Eigen::Vector3d(param_pose_[i][0], param_pose_[i][1], param_pose_[i][2]);
    }
  }

  Eigen::VectorXd dep = feature_manager_->getDepthVector();
  for (int i = 0; i < feature_manager_->getFeatureCount(); i++)
    dep(i) = param_feature_[i][0];
  feature_manager_->setDepth(dep);
}

}; // namespace estimator
