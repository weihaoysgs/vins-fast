#include "estimator/feature_manager.hpp"

namespace estimator {

FeatureManager::FeatureManager()
{
  MIN_PARALLAX = common::Setting::getSingleton()->Get<double>("keyframe_parallax");
  FOCAL_LENGTH = common::Setting::getSingleton()->Get<double>("focal_length");
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
  LOG(INFO) << "MIN_PARALLAX: " << MIN_PARALLAX;
  for (auto &ric : ric_)
    ric.setIdentity();
}

void FeatureManager::SetRic(Eigen::Matrix3d *ric)
{
  for (int i = 0; i < 2; i++)
  {
    ric_[i] = ric[i];
  }
}

/// @return true this frame is keyframe
/// @return false this frame is not keyframe
bool FeatureManager::AddFeatureAndCheckParallax(
    int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
  last_track_num_ = 0;
  int new_landmark_num = 0;
  int long_time_track_num = 0;
  double parallax_sum = 0;
  int parallax_num = 0;

  for (auto &id_pts : image)
  {
    ObservedFrameFeature observed_frame_feature = ObservedFrameFeature(id_pts.second[0].second, td);
    assert(id_pts.second.size() <= 2);
    assert(id_pts.second[0].first == 0);
    if (id_pts.second.size() == 2)
    {
      observed_frame_feature.RightObservation(id_pts.second[1].second);
      assert(id_pts.second[1].first == 1);
    }
    int feature_id = id_pts.first;
    auto it = std::find_if(features_.begin(), features_.end(), [feature_id](const IDWithObservedFeatures &it) {
      return it.feature_id_ == feature_id;
    });
    /// is a new landmark
    if (it == features_.end())
    {
      features_.push_back(IDWithObservedFeatures(feature_id, frame_count));
      features_.back().feature_per_frame_.push_back(observed_frame_feature);
      new_landmark_num++;
    }
    /// is a old tracking landmark
    else if (it->feature_id_ == feature_id)
    {
      it->feature_per_frame_.push_back(observed_frame_feature);
      last_track_num_++;
      if (it->feature_per_frame_.size() >= 4)
        long_time_track_num++;
    }
  }
  assert(last_track_num_ + new_landmark_num == image.size());
  /// the end condition is : when new feature extract larger than the MAX_PTS_EXTRACT_PER_FRAME
  if (frame_count < 2 || last_track_num_ < 20 || long_time_track_num < 40 || new_landmark_num > 0.5 * last_track_num_)
    return true;

  for (auto &id_with_observed : features_)
  {
    if (id_with_observed.start_frame_ <= frame_count - 2 && id_with_observed.EndFrame() >= frame_count - 1)
    {
      parallax_sum += ComputeParallax(id_with_observed, frame_count);
      parallax_num++;
    }
  }

  /// meaning that most feature can only observe by some new frame
  if (parallax_num == 0)
  {
    return true;
  }
  else
  {
    /// If the parallax is greater than or equal to the minimum parallax, it indicates that the movement is too fast, 
    /// the similarity between the two images is reduced, etc., and it should be listed as a key frame.
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

double FeatureManager::ComputeParallax(const IDWithObservedFeatures &it_per_id, int frame_count)
{
  // check the second last frame is keyframe or not
  // parallax betwwen seconde last frame and third last frame
  const ObservedFrameFeature &frame_i = it_per_id.feature_per_frame_[frame_count - 2 - it_per_id.start_frame_];
  const ObservedFrameFeature &frame_j = it_per_id.feature_per_frame_[frame_count - 1 - it_per_id.start_frame_];

  Eigen::Vector3d p_j = frame_j.point_;
  double u_j = p_j(0);
  double v_j = p_j(1);

  Eigen::Vector3d p_i = frame_i.point_;
  double u_i = p_i(0);
  double v_i = p_i(1);

  double du = u_i - u_j, dv = v_i - v_j;
  return sqrt(du * du + dv * dv);
}
/// @brief calculate the newest frame init pose using PNP
void FeatureManager::InitFramePoseByPnP(int frame_cnt, Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs, Eigen::Vector3d *tic,
                                        Eigen::Matrix3d *ric)
{
  if (frame_cnt <= 0)
    return;
  std::vector<cv::Point2f> pts2D;
  std::vector<cv::Point3f> pts3D;

  for (auto &id_observe_feature : features_)
  {
    double estimate_depth = id_observe_feature.estimated_depth_;
    if (estimate_depth > 0)
    {
      /// get the newest observation pts index in the [feature_per_frame_].
      int index = frame_cnt - id_observe_feature.start_frame_;
      /// only get the landmark which can be observe by newest frame
      if (id_observe_feature.EndFrame() >= frame_cnt)
      {
        Eigen::Vector3d pts_in_imu =
            ric[0] * (id_observe_feature.feature_per_frame_[0].point_ * estimate_depth) + tic[0];
        Eigen::Vector3d pts_in_world =
            Rs[id_observe_feature.start_frame_] * pts_in_imu + Ps[id_observe_feature.start_frame_];
        cv::Point3f point3d(pts_in_world.x(), pts_in_world.y(), pts_in_world.z());
        /// get this landmark observation in the newest frame
        cv::Point2f point2d(id_observe_feature.feature_per_frame_[index].point_.x(),
                            id_observe_feature.feature_per_frame_[index].point_.y());
        pts2D.push_back(point2d);
        pts3D.push_back(point3d);
      }
    }
  }

  Eigen::Matrix3d RCam;
  Eigen::Vector3d PCam;
  /// R_wc = R_wi * r_ic
  RCam = Rs[frame_cnt - 1] * ric[0];
  /// T_wc = R_wi * t_ic + P_wi
  PCam = Rs[frame_cnt - 1] * tic[0] + Ps[frame_cnt - 1];
  if (SolvePoseByPnP(RCam, PCam, pts2D, pts3D))
  {
    /// R_wc -> R_wi; P_wc -> P_wi
    Rs[frame_cnt] = RCam * ric[0].transpose();
    Ps[frame_cnt] = -RCam * ric[0].transpose() * tic[0] + PCam;
  }
  else
  {
    LOG(ERROR) << "Solve PNP Failed";
  }
}

bool FeatureManager::SolvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, std::vector<cv::Point2f> &pts2D,
                                    std::vector<cv::Point3f> &pts3D)
{
  Eigen::Matrix3d R_initial;
  Eigen::Vector3d P_initial;

  /// w_T_cam ---> cam_T_w
  /// R_wc -> R_cw; P_wc->P_cw
  R_initial = R.inverse();
  P_initial = -(R_initial * P);

  if (int(pts2D.size()) < 4)
  {
    LOG(ERROR) << "Feature tracking not enough, please slowly move you device!";
    return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);

  if (!pnp_succ)
  {
    LOG(ERROR) << "pnp failed !";
    return false;
  }
  cv::Rodrigues(rvec, r);
  Eigen::MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  Eigen::MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);

  /// cam_T_w ---> w_T_cam
  /// R_cw -> R_wc; P_cw->P_wc
  R = R_pnp.transpose();
  P = R * (-T_pnp);
  return true;
}

void FeatureManager::TriangulatePts(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[],
                                    Eigen::Matrix3d ric[])
{
  int right_observed_num = 0;
  for (auto &id_observed_feature : features_)
  {
    // LOG(INFO) << "observed times: " << id_observed_feature.feature_per_frame_.size();
    if (id_observed_feature.estimated_depth_ > 0)
      continue;
    /// if track success in right image. using the camera ext param to triangulate
    if (id_observed_feature.feature_per_frame_[0].track_right_success_)
    {
      right_observed_num++;
      /// triangulate frame index in window
      int window_index = id_observed_feature.start_frame_;
      Eigen::Matrix<double, 3, 4> left_pose; /// T_cw
      /// P_wc = P_wi + R_wi * t_ic
      Eigen::Vector3d t0 = Ps[window_index] + Rs[window_index] * tic[0];
      /// R_wc = R_wi * r_ic
      Eigen::Matrix3d R0 = Rs[window_index] * ric[0];
      left_pose.leftCols<3>() = R0.transpose();
      left_pose.rightCols<1>() = -R0.transpose() * t0;

      Eigen::Matrix<double, 3, 4> right_pose;
      Eigen::Vector3d t1 = Ps[window_index] + Rs[window_index] * tic[1];
      Eigen::Matrix3d R1 = Rs[window_index] * ric[1];
      right_pose.leftCols<3>() = R1.transpose();
      right_pose.rightCols<1>() = -R1.transpose() * t1;
      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      /// according to the first observation in the left and right image, triangulate the point
      point0 = id_observed_feature.feature_per_frame_[0].point_.head(2);
      point1 = id_observed_feature.feature_per_frame_[0].point_right_.head(2);
      TriangulateOnePoint(left_pose, right_pose, point0, point1, point3d);
      /// R_wc = R_wc * p_c + T_wc
      Eigen::Vector3d local_point = left_pose.leftCols<3>() * point3d + left_pose.rightCols<1>();
      double depth = local_point.z();
      if (depth > 0)
        id_observed_feature.estimated_depth_ = depth;
      else
        id_observed_feature.estimated_depth_ = INIT_DEPTH;
      continue;
    }
    /// Observed by more than or equal to two frames, but not observed by the camera on the right
    else if (id_observed_feature.feature_per_frame_.size() > 1)
    {
      int imu_i = id_observed_feature.start_frame_;
      Eigen::Matrix<double, 3, 4> left_pose;
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
      Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
      left_pose.leftCols<3>() = R0.transpose();
      left_pose.rightCols<1>() = -R0.transpose() * t0;

      imu_i++;
      Eigen::Matrix<double, 3, 4> right_pose;
      Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
      right_pose.leftCols<3>() = R1.transpose();
      right_pose.rightCols<1>() = -R1.transpose() * t1;

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = id_observed_feature.feature_per_frame_[0].point_.head(2);
      point1 = id_observed_feature.feature_per_frame_[1].point_.head(2);
      TriangulateOnePoint(left_pose, right_pose, point0, point1, point3d);
      Eigen::Vector3d localPoint;
      localPoint = left_pose.leftCols<3>() * point3d + left_pose.rightCols<1>();
      double depth = localPoint.z();
      if (depth > 0)
        id_observed_feature.estimated_depth_ = depth;
      else
        id_observed_feature.estimated_depth_ = INIT_DEPTH;
      continue;
    }
  }
  // LOG(INFO) << "right_observed_num: " << right_observed_num;
}

void FeatureManager::TriangulateOnePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                                         Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
  design_matrix.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
  design_matrix.row(2) = point1[0] * pose1.row(2) - pose1.row(0);
  design_matrix.row(3) = point1[1] * pose1.row(2) - pose1.row(1);
  Eigen::Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}
void FeatureManager::RemoveBack()
{
  for (auto it = features_.begin(), it_next = features_.begin(); it != features_.end(); it = it_next)
  {
    it_next++;
    if (it->start_frame_ != 0)
      it->start_frame_--;
    else
    {
      it->feature_per_frame_.erase(it->feature_per_frame_.begin());
      if (it->feature_per_frame_.size() == 0)
        features_.erase(it);
    }
  }
}

void FeatureManager::RemoveBackShiftDepth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P,
                                          const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P)
{
  for (auto it = features_.begin(), it_next = features_.begin(); it != features_.end(); it = it_next)
  {
    it_next++;

    if (it->start_frame_ != 0)
      it->start_frame_--;
    else
    {
      Eigen::Vector3d uv_i = it->feature_per_frame_[0].point_;
      it->feature_per_frame_.erase(it->feature_per_frame_.begin());
      if (it->feature_per_frame_.size() < 2)
      {
        features_.erase(it);
        continue;
      }
      else
      {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth_; /// P_c
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;   /// P_wi
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth_ = dep_j;
        else
          it->estimated_depth_ = INIT_DEPTH;
      }
    }
  }
}

void FeatureManager::RemoveFront(int frame_count, int WINDOW_SIZE)
{
  for (auto it = features_.begin(), it_next = features_.begin(); it != features_.end(); it = it_next)
  {
    it_next++;

    if (it->start_frame_ == frame_count)
    {
      it->start_frame_--;
    }
    else
    {
      int j = WINDOW_SIZE - 1 - it->start_frame_;
      if (it->EndFrame() < frame_count - 1)
        continue;
      it->feature_per_frame_.erase(it->feature_per_frame_.begin() + j);
      if (it->feature_per_frame_.size() == 0)
        features_.erase(it);
    }
  }
}

double FeatureManager::ReProjectionError(const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi,
                                         const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
                                         const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj,
                                         const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj, double depth,
                                         const Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj)
{
  Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
  Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
  Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}

void FeatureManager::OutliersRejection(std::set<int> &removeIndex, const Eigen::Matrix3d *Rs, const Eigen::Vector3d *Ps,
                                       const Eigen::Matrix3d *ric, const Eigen::Vector3d *tic)
{
  int feature_index = -1;
  for (auto &it_per_id : features_)
  {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
    if (it_per_id.used_num_ < 4)
      continue;
    feature_index++;
    int imu_i = it_per_id.start_frame_, imu_j = imu_i - 1;
    Eigen::Vector3d pts_i = it_per_id.feature_per_frame_[0].point_;
    double depth = it_per_id.estimated_depth_;
    for (auto &it_per_frame : it_per_id.feature_per_frame_)
    {
      imu_j++;
      if (imu_i != imu_j)
      {
        Eigen::Vector3d pts_j = it_per_frame.point_;
        double tmp_error = ReProjectionError(
            Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
      }
      // need to rewrite projecton factor.........
      if (it_per_frame.track_right_success_)
      {
        Eigen::Vector3d pts_j_right = it_per_frame.point_right_;
        if (imu_i != imu_j)
        {
          double tmp_error = ReProjectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
        }
        else
        {
          double tmp_error = ReProjectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
        }
      }
    }
    double ave_err = err / errCnt;
    if (ave_err * FOCAL_LENGTH > 3)
      removeIndex.insert(it_per_id.feature_id_);
  }
}

void FeatureManager::RemoveOutlier(std::set<int> &outlierIndex)
{
  std::set<int>::iterator itSet;
  for (auto it = features_.begin(), it_next = features_.begin(); it != features_.end(); it = it_next)
  {
    it_next++;
    int index = it->feature_id_;
    itSet = outlierIndex.find(index);
    if (itSet != outlierIndex.end())
    {
      features_.erase(it);
    }
  }
}

int FeatureManager::getTriangulatedLandmarkNum() const
{
  int num = std::count_if(features_.begin(), features_.end(), [](const IDWithObservedFeatures &fea) -> bool {
    return fea.estimated_depth_ >= 0;
  });
  return num;
}

int FeatureManager::getFeatureCount()
{
  int cnt = 0;
  for (auto &it : features_)
  {
    it.used_num_ = it.feature_per_frame_.size();
    if (it.used_num_ >= 4)
    {
      cnt++;
    }
  }
  return cnt;
}

Eigen::VectorXd FeatureManager::getDepthVector()
{
  Eigen::VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : features_)
  {
    it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
    if (it_per_id.used_num_ < 4)
      continue;
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth_;
  }
  return dep_vec;
}

void FeatureManager::setDepth(const Eigen::VectorXd &x)
{
  int feature_index = -1;
  for (auto &it_per_id : features_)
  {
    it_per_id.used_num_ = it_per_id.feature_per_frame_.size();
    if (it_per_id.used_num_ < 4)
      continue;

    it_per_id.estimated_depth_ = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth_ < 0)
    {
      it_per_id.solve_flag_ = 2;
    }
    else
    {
      it_per_id.solve_flag_ = 1;
    }
  }
}

} // namespace estimator
