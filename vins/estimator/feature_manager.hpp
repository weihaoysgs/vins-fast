//
// Created by weihao on 23-9-27.
//

#ifndef FEATURE_MANAGER_HPP
#define FEATURE_MANAGER_HPP

#include "estimator/id_features.hpp"
#include "common/parameter.hpp"
#include <list>
#include <algorithm>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <set>

namespace estimator {
class FeatureManager
{
public:
  FeatureManager();
  void SetRic(Eigen::Matrix3d ric[]);
  bool AddFeatureAndCheckParallax(int frame_count,
                                  const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                  double td);
  double ComputeParallax(const IDWithObservedFeatures &it_per_id, int frame_count);
  void InitFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[],
                          Eigen::Matrix3d ric[]);
  bool SolvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, std::vector<cv::Point2f> &pts2D,
                      std::vector<cv::Point3f> &pts3D);
  void TriangulatePts(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
  void TriangulateOnePoint(Eigen::Matrix<double, 3, 4> &, Eigen::Matrix<double, 3, 4> &, Eigen::Vector2d &point0,
                           Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
  void RemoveBack();
  void RemoveFront(int frame_count, int window_size);
  void RemoveBackShiftDepth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P, const Eigen::Matrix3d &new_R,
                            const Eigen::Vector3d &new_P);
  double ReProjectionError(const Eigen::Matrix3d &Ri,const  Eigen::Vector3d &Pi,const  Eigen::Matrix3d &rici,
                           const  Eigen::Vector3d &tici, const Eigen::Matrix3d &Rj,const  Eigen::Vector3d &Pj,
                           const  Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj, double depth,
                           const  Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj);
  void OutliersRejection(std::set<int> &removeIndex, const Eigen::Matrix3d *Rs, const Eigen::Vector3d *Ps,
                         const Eigen::Matrix3d *ric, const Eigen::Vector3d *tic);
  void RemoveOutlier(std::set<int> &outlier_index);
  const std::list<IDWithObservedFeatures> &getFeatures() const { return features_; }
  int getTriangulatedLandmarkNum() const
  {
    int num = std::count_if(features_.begin(), features_.end(), [](const IDWithObservedFeatures &fea) -> bool {
      return fea.estimated_depth_ >= 0;
    });
    return num;
  }
  int getRightObservedNumber() const
  {
    int num = 0;
    for (const auto &id_observed_fea : features_)
    {
      if (id_observed_fea.feature_per_frame_.back().track_right_success_)
      {
        num++;
      }
    }
    return num;
  }

public:
  double MIN_PARALLAX;
  double FOCAL_LENGTH = 460.0;
  const double INIT_DEPTH = 5.0;

private:
  int last_track_num_ = 0;
  std::list<IDWithObservedFeatures> features_;
  Eigen::Matrix3d ric_[2];
};

} // namespace estimator

#endif // FEATURE_MANAGER_HPP
