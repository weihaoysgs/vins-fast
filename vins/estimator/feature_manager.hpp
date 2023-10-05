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
  /// constructor fun
  FeatureManager();

  void SetRic(Eigen::Matrix3d ric[]);

  /// @brief Add new landmark to map, determine whether it is a key frame based on parallax
  bool AddFeatureAndCheckParallax(int frame_count,
                                  const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                  double td);

  /// @brief Compute parallax between second last frame and third last frame
  double ComputeParallax(const IDWithObservedFeatures &it_per_id, int frame_count);

  /// @brief Init the newest frame pose by PNP solver
  void InitFramePoseByPnP(int frameCnt, Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[],
                          Eigen::Matrix3d ric[]);

  /// @brief Solve PNP
  bool SolvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, std::vector<cv::Point2f> &pts2D,
                      std::vector<cv::Point3f> &pts3D);

  /// @brief Triangulate all pts in window
  void TriangulatePts(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

  /// @brief get triangulated pts inverse depth
  Eigen::VectorXd getDepthVector();

  void setDepth(const Eigen::VectorXd &x);

  /// @brief get number of userful landmarks
  int getFeatureCount();

  /// @brief Triangulate one point
  void TriangulateOnePoint(Eigen::Matrix<double, 3, 4> &, Eigen::Matrix<double, 3, 4> &, Eigen::Vector2d &point0,
                           Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);

  /// @brief Remove the landmark which is observed by first frame in window
  void RemoveBack();

  /// @brief Remove the landmark which is observed by [frame_count] frame in window
  void RemoveFront(int frame_count, int window_size);

  void RemoveBackShiftDepth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P, const Eigen::Matrix3d &new_R,
                            const Eigen::Vector3d &new_P);

  /// @brief Compute the re projection error
  double ReProjectionError(const Eigen::Matrix3d &Ri,const  Eigen::Vector3d &Pi,const  Eigen::Matrix3d &rici,
                           const  Eigen::Vector3d &, const Eigen::Matrix3d &Rj,const  Eigen::Vector3d &Pj,
                           const  Eigen::Matrix3d &, const Eigen::Vector3d &, double depth,
                           const  Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj);

  /// @brief Outliers rejection
  void OutliersRejection(std::set<int> &removeIndex, const Eigen::Matrix3d *Rs, const Eigen::Vector3d *Ps,
                         const Eigen::Matrix3d *ric, const Eigen::Vector3d *tic);

  /// @brief Remove outliers
  void RemoveOutlier(std::set<int> &outlier_index);

  /// @brief Get all landmark in window
  std::list<IDWithObservedFeatures> &getFeatures() { return features_; }

  int getTriangulatedLandmarkNum() const;

public:
  double MIN_PARALLAX;
  double FOCAL_LENGTH = 460.0;
  const double INIT_DEPTH = 5.0;

private:
  std::list<IDWithObservedFeatures> features_;
  int last_track_num_ = 0;
  Eigen::Matrix3d ric_[2];
};

} // namespace estimator

#endif // FEATURE_MANAGER_HPP
