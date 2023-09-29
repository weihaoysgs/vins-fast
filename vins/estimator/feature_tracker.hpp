#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include <map>
#include <vector>
#include <cassert>
#include <mutex>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include "common/parameter.hpp"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

namespace estimator {

class FeatureTracker
{
public:
  /// @brief constructor function
  FeatureTracker();

  /// @brief Main process image fun
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
  TrackImage(double current_time, const cv::Mat &img0, const cv::Mat &img1 = cv::Mat());

  /// @brief Check the pts in image
  bool InBorder(const cv::Point2f &pt, int row, int col);

  /// @brief Remove the distortion
  std::vector<cv::Point2f> RemoveDistortion(std::vector<cv::Point2f> &src_pts, camodocal::CameraPtr cam);

  /// @brief Compute the kps velocity
  std::vector<cv::Point2f> ComputeKpsVelocity(std::vector<int> &ids, std::vector<cv::Point2f> &un_distortion_kps,
                                              std::map<int, cv::Point2f> &current_id_pts,
                                              std::map<int, cv::Point2f> &previous_id_pts);

  /// @brief Draw the kps velocity
  void DrawVelocity(const cv::Mat &img, const std::vector<int> &ids, const std::vector<int> &track_count,
                    std::vector<cv::Point2f> &kps_velocity, std::vector<cv::Point2f> &kps, const std::string &name);

  /// @brief Draw tracker result
  void DrawTracker();

  /// @brief Show un distortion result, the calculation is complex
  void ShowUnDistortion();

  /// @brief Get current image, TODO: not using
  cv::Mat getCurrentImage() const { return current_img_; }

  /// @brief Get previous image, TODO: not using
  cv::Mat getPreviousImage() const { return previous_img_; }

  /// @brief Read camera intrinsic param
  void ReadIntrinsicParameter(const std::vector<std::string> &calib_file_path);

  /// @brief Draw image track count num
  void DrawIdsTrackCount(int wait_key);

  /// @brief Draw track result with line
  void DrawTrackResultWithLine(int wait_key);

  /// @brief Set feature extractor mask
  void SetFeatureExtractorMask(cv::Mat &mask);

  /// @brief Get draw tracker result image to show
  cv::Mat getDrawTrackResultImg();

  /// @brief Compute distance between two cv points
  double PtDistance(cv::Point2f &pt1, cv::Point2f &pt2) const;

  /// @brief Template fun, reduce the vector according the status
  template <typename T, typename S> void ReduceVector(std::vector<T> &v, const std::vector<S> &status)
  {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
      if (status[i])
        v[j++] = v[i];
    v.resize(j);
  }

public:
  int OPTICAL_FLOW_BACK;
  int MAX_CNT_FEATURES_PER_FRAME = 0;
  int MIN_DIST; /// min distance between two features
  int SHOW_TRACK_RESULT;
  int FRONTEND_WAIT_KEY;
public:
  std::mutex draw_track_img_mutex_;

private:
  unsigned int landmark_id_ = 0;
  double current_time_ = 0, previous_time_ = 0;

  std::vector<camodocal::CameraPtr> cameras_;
  std::vector<int> ids_, ids_right_;
  std::vector<int> track_count_;
  std::vector<cv::Point2f> previous_kps_, current_kps_, current_right_img_kps_;
  std::vector<cv::Point2f> current_un_distortion_kps_, current_right_un_distortion_kps_;
  std::vector<cv::Point2f> pts_velocity, right_pts_velocity;
  std::map<int, cv::Point2f> previous_left_ids_kps_map_; /// only use to draw track img
  std::map<int, cv::Point2f> previous_left_un_distortion_ids_kps_map_, previous_right_un_distortion_ids_kps_map_;
  std::map<int, cv::Point2f> current_left_un_distortion_ids_kps_map_, current_right_un_distortion_ids_kps_map_;
  cv::Mat current_img_, current_right_img_, previous_img_;
  cv::Mat draw_track_img_result_;
};

} // namespace estimator

#endif // FEATURE_TRACKER_HPP
