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
  SHOW_TRACK_RESULT = common::Setting::getSingleton()->Get<int>("show_track");
  FRONTEND_WAIT_KEY = common::Setting::getSingleton()->Get<int>("frontend_wait_key");
  LOG(INFO) << "MAX_CNT_FEATURES_PER_FRAME: " << MAX_CNT_FEATURES_PER_FRAME;
  LOG(INFO) << "OPTICAL_FLOW_BACK: " << OPTICAL_FLOW_BACK;
  LOG(INFO) << "MIN_DIST: " << MIN_DIST;
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
    // size_t track_success_cnt = std::count(status.begin(), status.end(), true);
    // LOG(INFO) << "Before back track: " << track_success_cnt;
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
      for (size_t i = 0; i < status.size(); i++)
      {
        if (!(status[i] && reverse_status[i] && PtDistance(previous_kps_[i], reverse_pts[i]) <= 0.5))
          status[i] = 0;
      }
    }
    // track_success_cnt = std::count(status.begin(), status.end(), true);
    // LOG(INFO) << "After back track: " << track_success_cnt;
    for (int i = 0; i < int(current_kps_.size()); i++)
    {
      if (status[i] && !InBorder(current_kps_[i], current_img_.rows, current_img_.cols))
        status[i] = 0;
    }
    // DrawTrackResultWithLine(1);
    ReduceVector(current_kps_, status);
    ReduceVector(ids_, status);
    ReduceVector(track_count_, status);
    // DrawIdsTrackCount(1);
  }

  /// if track success, add the track cnt
  for (auto &n : track_count_)
  {
    n++;
  }

  /// extract new feature
  cv::Mat mask;
  /// set mask will reduce a little current kps
  SetFeatureExtractorMask(mask);
  int new_feature_tobe_extract = MAX_CNT_FEATURES_PER_FRAME - static_cast<int>(current_kps_.size());

  std::vector<cv::Point2f> new_feature_pts;
  if (new_feature_tobe_extract > 0)
  {
    cv::goodFeaturesToTrack(current_img_, new_feature_pts, new_feature_tobe_extract, 0.01, MIN_DIST, mask);
  }

  // assert(new_feature_tobe_extract + current_kps_.size() == MAX_CNT_FEATURES_PER_FRAME);

  /// add new feature
  for (auto &pt : new_feature_pts)
  {
    current_kps_.emplace_back(pt);
    track_count_.emplace_back(1);
    ids_.emplace_back(landmark_id_++);
  }

  current_un_distortion_kps_ = RemoveDistortion(current_kps_, cameras_[0]);
  pts_velocity = ComputeKpsVelocity(ids_,
                                    current_un_distortion_kps_,
                                    current_left_un_distortion_ids_kps_map_,
                                    previous_left_un_distortion_ids_kps_map_);

  /// tracking right image using left image kps
  {
    // clang-format off
    ids_right_.clear();
    current_right_img_kps_.clear();
    std::vector<float> err;
    std::vector<uchar> status;

    cv::calcOpticalFlowPyrLK(current_img_, current_right_img_, current_kps_,
                             current_right_img_kps_, status, err,
                              cv::Size(21, 21), 3);
    if (OPTICAL_FLOW_BACK)
    {
      std::vector<cv::Point2f> reverse_left_kps;
      std::vector<float> err_right;
      std::vector<uchar> status_right;
      cv::calcOpticalFlowPyrLK(current_right_img_, current_img_, current_right_img_kps_,
                        reverse_left_kps, status_right, err_right,
                        cv::Size(21, 21), 3);
      for(size_t i = 0; i < status.size(); i++)
      {
          if(!(status[i] && status_right[i] && InBorder(current_right_img_kps_[i], current_right_img_.rows,
             current_right_img_.cols) && PtDistance(current_kps_[i], reverse_left_kps[i]) <= 0.5))
            status[i] = 0;
      }
    }

    ids_right_ = ids_;
    ReduceVector(current_right_img_kps_, status);
    ReduceVector(ids_right_, status);
    current_right_un_distortion_kps_ = RemoveDistortion(current_right_img_kps_, cameras_[1]);
    right_pts_velocity = ComputeKpsVelocity(ids_right_, current_right_un_distortion_kps_,
                        current_right_un_distortion_ids_kps_map_, previous_right_un_distortion_ids_kps_map_);
    // clang-format on
  }

  if (SHOW_TRACK_RESULT)
  {
    {
      std::unique_lock<std::mutex> lck(draw_track_img_mutex_);
      DrawTracker();
    }
    if (FRONTEND_WAIT_KEY == 0)
    {
      cv::imshow("draw tracker", draw_track_img_result_);
      cv::waitKey(FRONTEND_WAIT_KEY);
    }
  }

  previous_left_ids_kps_map_.clear();
  for (size_t i = 0; i < current_kps_.size(); i++)
  {
    previous_left_ids_kps_map_[ids_[i]] = current_kps_[i];
  }

  previous_img_ = current_img_;
  previous_kps_ = current_kps_;
  previous_left_un_distortion_ids_kps_map_ = current_left_un_distortion_ids_kps_map_;
  previous_right_un_distortion_ids_kps_map_ = current_right_un_distortion_ids_kps_map_;
  previous_time_ = current_time_;

  // DrawVelocity(current_img_, ids_, track_count_, pts_velocity, current_kps_, "left vel");
  // DrawVelocity(current_right_img_, ids_right_, track_count_, right_pts_velocity, current_right_img_kps_, "right vel");

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> feature_frame;

  for (size_t i = 0; i < ids_.size(); i++)
  {
    int feature_id = ids_[i];
    double x, y, z;
    x = current_un_distortion_kps_[i].x;
    y = current_un_distortion_kps_[i].y;
    z = 1;
    double p_u, p_v;
    p_u = current_kps_[i].x;
    p_v = current_kps_[i].y;
    int camera_id = 0;
    double velocity_x, velocity_y;
    velocity_x = pts_velocity[i].x;
    velocity_y = pts_velocity[i].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  for (size_t i = 0; i < ids_right_.size(); i++)
  {
    int feature_id = ids_right_[i];
    double x, y, z;
    x = current_right_un_distortion_kps_[i].x;
    y = current_right_un_distortion_kps_[i].y;
    z = 1;
    double p_u, p_v;
    p_u = current_right_img_kps_[i].x;
    p_v = current_right_img_kps_[i].y;
    int camera_id = 1;
    double velocity_x, velocity_y;
    velocity_x = right_pts_velocity[i].x;
    velocity_y = right_pts_velocity[i].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    feature_frame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }
  // ShowUnDistortion();

  return feature_frame;
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
      current_kps_.push_back(it.second.first);
      ids_.push_back(it.second.second);
      track_count_.push_back(it.first);
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

void FeatureTracker::DrawTracker()
{
  // clang-format off
  auto cols = static_cast<float>(current_img_.cols);
  cv::hconcat(current_img_, current_right_img_, draw_track_img_result_);
  cv::cvtColor(draw_track_img_result_, draw_track_img_result_, CV_GRAY2RGB);
  for (size_t j = 0; j < current_kps_.size(); j++)
  {
    double len = std::min(1.0, 1.0 * track_count_[j] / 20);
    /// len = 0 is blue， len = 1 is red
    cv::circle(draw_track_img_result_, current_kps_[j], 2,
               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  for (const auto &kp : current_right_img_kps_)
  {
    cv::Point2f right_kp = kp;
    right_kp.x += cols;
    cv::circle(draw_track_img_result_, right_kp, 2,
               cv::Scalar(0, 255, 0), 2);
  }
  for (size_t i = 0; i < current_kps_.size(); i++)
  {
    auto id = ids_[i];
    auto iter = previous_left_ids_kps_map_.find(id);
    if (iter != previous_left_ids_kps_map_.end())
    {
      cv::arrowedLine(draw_track_img_result_, current_kps_[i], iter->second,
                      cv::Scalar(0, 255, 0), 1,
                      8, 0, 0.2);
    }
  }
  // clang-format on
}

void FeatureTracker::ReadIntrinsicParameter(const std::vector<std::string> &calib_file_path)
{
  for (const auto &path : calib_file_path)
  {
    camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(path);
    cameras_.push_back(camera);
  }
}

std::vector<cv::Point2f> FeatureTracker::RemoveDistortion(std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
  std::vector<cv::Point2f> un_distortion_kps;
  for (auto &pt : pts)
  {
    Eigen::Vector2d a(pt.x, pt.y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_distortion_kps.emplace_back(b.x() / b.z(), b.y() / b.z());
  }
  return un_distortion_kps;
}

std::vector<cv::Point2f> FeatureTracker::ComputeKpsVelocity(std::vector<int> &ids,
                                                            std::vector<cv::Point2f> &un_distortion_kps,
                                                            std::map<int, cv::Point2f> &current_id_pts,
                                                            std::map<int, cv::Point2f> &previous_id_pts)
{
  assert(ids.size() == un_distortion_kps.size());
  std::vector<cv::Point2f> pts_velocity;
  current_id_pts.clear();

  for (unsigned int i = 0; i < ids.size(); i++)
  {
    current_id_pts.insert(std::make_pair(ids[i], un_distortion_kps[i]));
  }

  if (!previous_id_pts.empty())
  {
    double dt = current_time_ - previous_time_;

    for (unsigned int i = 0; i < un_distortion_kps.size(); i++)
    {
      std::map<int, cv::Point2f>::iterator it;
      it = previous_id_pts.find(ids[i]);
      if (it != previous_id_pts.end())
      {
        double v_x = (un_distortion_kps[i].x - it->second.x) / dt;
        double v_y = (un_distortion_kps[i].y - it->second.y) / dt;
        pts_velocity.emplace_back(v_x, v_y);
      }
      else
        pts_velocity.emplace_back(0, 0);
    }
  }
  else
  {
    for (unsigned int i = 0; i < un_distortion_kps.size(); i++)
    {
      pts_velocity.emplace_back(0, 0);
    }
  }
  return pts_velocity;
}

void FeatureTracker::ShowUnDistortion()
{
  const size_t col = current_img_.cols;
  const size_t row = current_img_.rows;

  cv::Mat undistorted_img(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
  std::vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < col; i++)
    for (int j = 0; j < row; j++)
    {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      cameras_[0]->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++)
  {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * 460.0 + col / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * 460.0 + row / 2;
    pp.at<float>(2, 0) = 1.0;

    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 &&
        pp.at<float>(0, 0) + 300 < col + 600)
    {
      undistorted_img.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
          current_img_.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
  }
  // turn the following code on if you need
  cv::imshow("un_distortion", undistorted_img);
}

void FeatureTracker::DrawVelocity(const cv::Mat &img, const std::vector<int> &ids, const std::vector<int> &track_count,
                                  std::vector<cv::Point2f> &kps_velocity, std::vector<cv::Point2f> &kps,
                                  const std::string &name)
{
  // clang-format off
  const int expand_size = 2;
  cv::Mat color_img, draw_img;
  cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);
  cv::resize(color_img, draw_img, cv::Size(color_img.cols * expand_size, color_img.rows * expand_size));
  for (size_t i = 0; i < ids.size(); i++)
  {
    int feature_id = ids[i];
    int cnt = track_count[i];
    cv::Point2f pt = kps[i] * expand_size;
    double vel_x = kps_velocity[i].x;
    double vel_y = kps_velocity[i].y;

    cv::circle(draw_img, pt, 2, cv::Scalar(0, 255, 0), -1);
    cv::putText(draw_img, "id: " + std::to_string(feature_id), pt - cv::Point2f(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
    cv::putText(draw_img, "cnt: " + std::to_string(cnt), pt - cv::Point2f(10, 0),
            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
    cv::putText(draw_img, std::string("v: " + std::to_string(vel_x).substr(0, 3)
                  + "," + std::to_string(vel_y).substr(0, 3)),
                pt + cv::Point2f(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.3,
                cv::Scalar(0, 255, 0), 1);
  }
  cv::imshow(name, draw_img);
  // clang-format on
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
cv::Mat FeatureTracker::getDrawTrackResultImg()
{
  std::unique_lock<std::mutex> lck(draw_track_img_mutex_);
  return draw_track_img_result_;
}

} // namespace estimator