//
// Created by weihao on 23-9-26.
//

#include "feature_tracker.hpp"

namespace estimator {
FeatureTracker::FeatureTracker()
{
  MAX_CNT_FEATURES_PER_FRAME = common::Setting::getSingleton()->Get<int>("max_cnt");
  OPTICAL_FLOW_BACK = common::Setting::getSingleton()->Get<int>("flow_back");
  MIN_DIST = common::Setting::getSingleton()->Get<int>("min_dist");
  LOG(INFO) << "MAX_CNT_FEATURES_PER_FRAME: " << MAX_CNT_FEATURES_PER_FRAME;
  LOG(INFO) << "OPTICAL_FLOW_BACK: " << OPTICAL_FLOW_BACK;
}

bool FeatureTracker::InBorder(const cv::Point2f &pt, int row, int col)
{
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return 1 <= img_x && img_x < col - 1 && 1 <= img_y && img_y < row - 1;
}

std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::TrackImage(double current_time, const cv::Mat &img0, const cv::Mat &img1)
{
  assert(!img0.empty() && !img1.empty());
  current_time_ = current_time;
  current_img_ = img0;
  current_right_img_ = img1;
  current_kps_.clear();

  /// tracking, but only tracking left image
  if (!previous_kps_.empty())
  {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        previous_img_, current_img_, previous_kps_, current_kps_, status, err, cv::Size(21, 21), 3);
    if (OPTICAL_FLOW_BACK)
    {
      // clang-format off
      std::vector<uchar> reverse_status;
      std::vector<cv::Point2f> reverse_pts = previous_kps_;
      cv::calcOpticalFlowPyrLK(current_img_, previous_img_, current_kps_, reverse_pts,
                               reverse_status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                               cv::OPTFLOW_USE_INITIAL_FLOW);
      // clang-format on
      for (size_t i = 0; i < status[i]; i++)
      {
        if (status[i] && reverse_status[i] && PtDistance(previous_kps_[i], reverse_pts[i]) <= 0.5)
        {
          status[i] = 1;
        }
        else
        {
          status[i] = 0;
        }
      }
    }
    for (int i = 0; i < int(current_kps_.size()); i++)
    {
      if (status[i] && !InBorder(current_kps_[i], current_img_.rows, current_img_.cols))
        status[i] = 0;
    }
    DrawTrackResultWithLine(1);
    ReduceVector(current_kps_, status);
    ReduceVector(ids_, status);
    ReduceVector(track_count_, status);
    DrawIdsTrackCount(0);
  }

  /// if track success, add the track cnt
  for (auto &n : track_count_)
  {
    n++;
  }

  /// extract new feature
  int new_feature_tobe_extract = MAX_CNT_FEATURES_PER_FRAME - static_cast<int>(current_kps_.size());
  std::vector<cv::Point2f> new_feature_pts;
  if (new_feature_tobe_extract > 0)
  {
    cv::Mat mask;
    SetFeatureExtractorMask(mask);
    cv::goodFeaturesToTrack(current_img_, new_feature_pts, new_feature_tobe_extract, 0.01, MIN_DIST, mask);
  }

  /// add new feature
  for (auto &pt : new_feature_pts)
  {
    current_kps_.emplace_back(pt);
    track_count_.emplace_back(1);
    ids_.emplace_back(landmark_id_++);
  }

  previous_img_ = current_img_;
  previous_kps_ = current_kps_;

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> feature_frame;

  return feature_frame;
}

void FeatureTracker::DrawIdsTrackCount(int wait_key)
{
  // clang-format off
  cv::Mat draw_img;
  cv::cvtColor(current_img_, draw_img, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < current_kps_.size(); ++i)
  {
    cv::Point2f pt = current_kps_[i];
    int id = ids_[i];
    int cnt = track_count_[i];
    cv::circle(draw_img, pt, 1, cv::Scalar(0, 0, 255), -1);
    cv::putText(draw_img, "id: " + std::to_string(id), pt - cv::Point2f(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
    cv::putText(draw_img, "cnt: " + std::to_string(cnt), pt + cv::Point2f(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
  }
  cv::imshow("track: ", draw_img);
  cv::waitKey(wait_key);
  // clang-format on
}

void FeatureTracker::SetFeatureExtractorMask(cv::Mat &mask)
{
  mask = cv::Mat(current_img_.rows, current_img_.cols, CV_8UC1, cv::Scalar(255));
  /// prefer to keep features that are tracked for long time
  std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
  for (unsigned int i = 0; i < current_kps_.size(); i++)
  {
    cnt_pts_id.emplace_back(track_count_[i], std::make_pair(current_kps_[i], ids_[i]));
  }
  std::sort(cnt_pts_id.begin(),
            cnt_pts_id.end(),
            [](const std::pair<int, std::pair<cv::Point2f, int>> &a,
               const std::pair<int, std::pair<cv::Point2f, int>> &b) { return a.first > b.first; });
  current_kps_.clear();
  ids_.clear();
  track_count_.clear();

  for (auto &it : cnt_pts_id)
  {
    if (mask.at<uchar>(it.second.first) == 255)
    {
      current_kps_.emplace_back(it.second.first);
      ids_.emplace_back(it.second.second);
      track_count_.emplace_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

void FeatureTracker::DrawTrackResultWithLine(int wait_key)
{
  cv::Mat draw_img;
  cv::cvtColor(previous_img_, draw_img, cv::COLOR_GRAY2BGR);
  assert(previous_kps_.size() == current_kps_.size());
  for (size_t i = 0; i < current_kps_.size(); ++i)
  {
    cv::Point2f pt1 = current_kps_[i];
    if (pt1.x >= 0 && pt1.y >= 0)
    {
      cv::Point2f pt2 = previous_kps_[i];
      cv::circle(draw_img, pt1, 2, cv::Scalar(0, 255, 0), -1);
      cv::line(draw_img, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }
  }
  cv::imshow("track_result", draw_img);
  cv::waitKey(wait_key);
}
} // namespace estimator