#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include <map>
#include <vector>
#include <cassert>
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
  FeatureTracker();
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
  TrackImage(double current_time, const cv::Mat &img0, const cv::Mat &img1 = cv::Mat());
  bool InBorder(const cv::Point2f &pt, int row, int col);
  std::vector<cv::Point2f> RemoveDistortion(std::vector<cv::Point2f> &src_pts, camodocal::CameraPtr cam);
  std::vector<cv::Point2f> ComputeKpsVelocity(std::vector<int> &ids, std::vector<cv::Point2f> &un_distortion_kps,
                                              std::map<int, cv::Point2f> &current_id_pts,
                                              std::map<int, cv::Point2f> &previous_id_pts);
  void DrawVelocity(const cv::Mat &img, const std::vector<int> &ids, const std::vector<int> &track_count,
                    std::vector<cv::Point2f> &kps_velocity, std::vector<cv::Point2f> &kps, const std::string &name);
  void DrawTracker();
  void ShowUnDistortion();
  void DrawVelocityLeft();
  void DrawVelocityRight();
  void ReadIntrinsicParameter(const std::vector<std::string> &calib_file_path);
  void DrawIdsTrackCount(int wait_key);
  void DrawTrackResultWithLine(int wait_key);
  void SetFeatureExtractorMask(cv::Mat &mask);
  cv::Mat getDrawTrackResultImg() const { return draw_track_img_result_; }
  double PtDistance(cv::Point2f &pt1, cv::Point2f &pt2)
  {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
  }

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
