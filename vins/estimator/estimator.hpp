#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include "estimator/feature_tracker.hpp"
#include "estimator/feature_manager.hpp"
#include "common/utils.hpp"
#include "factor/pose_local_parameterization.hpp"
#include "factor/projection2frame1camera_factor.hpp"
#include "factor/projection2frame2camera_factor.hpp"
#include "factor/projection1frame2camera_factor.hpp"
#include "factor/marginalization/marg_factor.hpp"
#include "common/size_pose_param.hpp"
#include "Eigen/Core"
#include "common/visualization.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "Eigen/Dense"
#include <mutex>
#include <ceres/problem.h>
#include <sensor_msgs/Image.h>
#include <queue>
#include <thread>
#include "common/time_calculate.hpp"

namespace estimator {

class Estimator
{
public:
  enum SolverFlag
  {
    INITIAL,
    NON_LINEAR
  };

  enum MarginalizationFlag
  {
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
  };

public:
  Estimator();
  void tFrontendProcess();
  void tBackendProcess();
  void FrontendTracker(double t, const cv::Mat &img0, const cv::Mat &img1);
  void ReadImuCameraExternalParam();
  void ProcessImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                    const double time);
  void ClearState();
  void SetParameter();
  void Optimization();
  void PrepareMarginalizationFactor();
  void SlideWindow();
  void SlideWindowOld();
  void SlideWindowNew();
  void vector2double();
  void double2vector();
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
  MarginalizationFlag marg_flag_;
  SolverFlag solver_flag_;
  factor::MarginalizationInfo *last_marginalization_info_ = nullptr;
  std::vector<double *> last_marginalization_parameter_blocks_;

  std::shared_ptr<FeatureTracker> feature_tracker_;
  std::shared_ptr<FeatureManager> feature_manager_;
  std::shared_ptr<common::Visualization> visualization_;

  static const int WINDOW_SIZE = 10;
  unsigned int input_image_cnt_{0};
  int frame_count_ = 0;

  /// time:feature_per_frame
  std::queue<std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>>> feature_buf_;
  std::queue<sensor_msgs::ImageConstPtr> img0_buf_, img1_buf_;

  /// R_{imu,camera},t_{imu,camera}
  Eigen::Matrix3d ric_[2];
  Eigen::Vector3d tic_[2];
  Eigen::Vector3d Ps_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Vs_[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d Rs_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Bas_[(WINDOW_SIZE + 1)];
  Eigen::Vector3d Bgs_[(WINDOW_SIZE + 1)];
  Eigen::Matrix3d back_R0_;
  Eigen::Vector3d back_P0_;
  double headers_[(WINDOW_SIZE + 1)];
  double time_diff_ = 0;
  double current_time_, previous_time_;

  double param_pose_[WINDOW_SIZE + 1][SIZE_POSE];
  double param_feature_[1000][SIZE_FEATURE];
  double param_ex_pose_[2][SIZE_POSE];
  double param_td_[1][1];

  cv::Mat current_img_;
public:
  std::mutex img_buf_mutex_;
  std::mutex feature_buf_mutex_;
  std::mutex current_img_mutex_;

public:
  int USE_IMU = 0;
  std::string IMG0_TOPIC_NAME, IMG1_TOPIC_NAME, IMU_TOPIC_NAME;
  std::vector<std::string> camera_calib_file_path_;
};

}; // namespace estimator

#endif // ESTIMATOR_HPP