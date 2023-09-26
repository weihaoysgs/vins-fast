#include "estimator/estimator.hpp"

namespace estimator {

};
estimator::Estimator::Estimator()
{
  USE_IMU = common::Setting::getSingleton()->Get<int>("imu");
  IMG0_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image0_topic");
  IMG1_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("image1_topic");
  IMU_TOPIC_NAME = common::Setting::getSingleton()->Get<std::string>("imu_topic");
  LOG(INFO)  << "IMG1 TOPIC: " << IMG1_TOPIC_NAME << ", IMG0 TOPIC: " << IMG1_TOPIC_NAME << ", IMU_TOPIC_NAME: " << IMU_TOPIC_NAME << ", USE_IMU:" << USE_IMU;

  feature_tracker_ = std::make_shared<estimator::FeatureTracker>();
  std::thread t_frontend_process = std::thread(&Estimator::tFrontendProcess, this);
  t_frontend_process.detach();
}

void estimator::Estimator::tFrontendProcess()
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
    if(!image0.empty() && !image1.empty())
    {
      ProcessImage(time, image0, image1);
    }
  }
}

void estimator::Estimator::ProcessImage(double t, const cv::Mat &img0, const cv::Mat &img1)
{
  input_image_cnt_++;
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> frame_feature;
  frame_feature = feature_tracker_->TrackImage(t, img0, img1);
  if(input_image_cnt_ % 2 == 0)
  {
    std::unique_lock<std::mutex> lck(feature_buf_mutex_);
    feature_buf_.push(std::make_pair(t, frame_feature));
  }
}
