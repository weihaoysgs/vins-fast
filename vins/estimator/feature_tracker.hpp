#ifndef FEATURETRACKER_HPP
#define FEATURETRACKER_HPP

#include <map>
#include <vector>
#include <cassert>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include "common/parameter.hpp"

namespace estimator {

class FeatureTracker
{
public:
  FeatureTracker();
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
  TrackImage(double current_time, const cv::Mat &img0, const cv::Mat &img1 = cv::Mat());
  bool InBorder(const cv::Point2f &pt, int row, int col);
  void DrawIdsTrackCount(int wait_key);
  void SetFeatureExtractorMask(cv::Mat &mask);
  double PtDistance(cv::Point2f &pt1, cv::Point2f &pt2)
  {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
  }

  template <typename T, typename S>
  void ReduceVector(std::vector<T> &v, const std::vector<S> &status)
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

private:
  double current_time_;
  std::vector<int> ids_, ids_right_;
  unsigned int landmark_id_ = 0;
  std::vector<int> track_count_;
  cv::Mat current_img_, current_right_img_, previous_img_;
  std::vector<cv::Point2f> previous_kps_, current_kps_, current_right_img_pts_;
};

} // namespace estimator

#endif //SRC_FEATURETRACKER_HPP
