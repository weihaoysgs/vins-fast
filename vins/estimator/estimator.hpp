#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include "estimator/feature_tracker.hpp"
#include "common/utils.hpp"
#include <mutex>
#include <sensor_msgs/Image.h>
#include <queue>
#include <thread>

namespace estimator {

class Estimator
{
public:
  Estimator();
  void tFrontendProcess();
  void ProcessImage(double t, const cv::Mat &img0, const cv::Mat &img1);

  void Image0Callback(const sensor_msgs::ImageConstPtr &img_msg)
  {
    std::unique_lock<std::mutex> lck(img_buf_mutex_);
    img0_buf_.push(img_msg);
  };

  void Image1Callback(const sensor_msgs::ImageConstPtr &img_msg)
  {
    std::unique_lock<std::mutex> lck(img_buf_mutex_);
    img1_buf_.push(img_msg);
  };

private:
  std::shared_ptr<estimator::FeatureTracker> feature_tracker_;
  unsigned int input_image_cnt_{0};
  /// time:feature_per_frame
  std::queue<std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>>> feature_buf_;
  std::queue<sensor_msgs::ImageConstPtr> img0_buf_, img1_buf_;

public:
  std::mutex img_buf_mutex_;
  std::mutex feature_buf_mutex_;
public:
  int USE_IMU = 0;
  std::string IMG0_TOPIC_NAME, IMG1_TOPIC_NAME, IMU_TOPIC_NAME;
};

}; // namespace estimator

#endif // ESTIMATOR_HPP