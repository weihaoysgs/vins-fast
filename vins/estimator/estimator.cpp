#include "estimator/estimator.hpp"

namespace estimator {

void Estimator::ReadImuCameraExternalParam()
{
  cv::Mat cv_T_0, cv_T_1;
  Eigen::Matrix4d T0, T1;
  common::Setting::getSingleton()->getFile()["body_T_cam0"] >> cv_T_0;
  cv::cv2eigen(cv_T_0, T0);
  ric[0] = T0.block<3, 3>(0, 0);
  tic[0] = T0.block<3, 1>(0, 3);
  common::Setting::getSingleton()->getFile()["body_T_cam1"] >> cv_T_1;
  cv::cv2eigen(cv_T_1, T1);
  ric[1] = T1.block<3, 3>(0, 0);
  tic[1] = T1.block<3, 1>(0, 3);
  LOG(INFO) << "Cam0 R:\n" << ric[0] << "\n t: " << tic[0].transpose();
  LOG(INFO) << "Cam1 R:\n" << ric[1] << "\n t: " << tic[1].transpose();

  std::string cam0_calib_file_path = common::Setting::getSingleton()->Get<std::string>("cam0_calib");
  std::string cam1_calib_file_path = common::Setting::getSingleton()->Get<std::string>("cam1_calib");
  assert(common::FileExists(cam0_calib_file_path) && common::FileExists(cam1_calib_file_path));
  camera_calib_file_path_.emplace_back(cam0_calib_file_path);
  camera_calib_file_path_.emplace_back(cam1_calib_file_path);
  LOG(INFO) << "Cam0 calib file: " << cam0_calib_file_path;
  LOG(INFO) << "Cam1 calib file: " << cam1_calib_file_path;
}

Estimator::Estimator()
{
  USE_IMU = common::Setting::getSingleton()->Get<int>("imu");
  IMG0_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image0_topic");
  IMG1_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image1_topic");
  IMU_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("imu_topic");
  LOG(INFO) << "IMG1 TOPIC: " << IMG1_TOPIC_NAME << ", IMG0 TOPIC: " << IMG1_TOPIC_NAME
            << ", IMU_TOPIC_NAME: " << IMU_TOPIC_NAME << ", USE_IMU:" << USE_IMU;

  ReadImuCameraExternalParam();
  feature_tracker_ = std::make_shared<estimator::FeatureTracker>();
  feature_tracker_->ReadIntrinsicParameter(camera_calib_file_path_);

  std::thread t_frontend_process = std::thread(&Estimator::tFrontendProcess, this);
  t_frontend_process.detach();
  std::thread t_backend_process = std::thread(&Estimator::tBackendProcess, this);
  t_backend_process.detach();
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
      ProcessImage(time, image0, image1);
    }
  }
}

void Estimator::ProcessImage(double t, const cv::Mat &img0, const cv::Mat &img1)
{
  input_image_cnt_++;
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
    LOG(INFO) <<"Backend";
    std::chrono::milliseconds dura(200);
    std::this_thread::sleep_for(dura);
  }
}

}; // namespace estimator