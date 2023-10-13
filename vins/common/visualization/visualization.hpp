#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "common/visualization/cam_pose_visualization.hpp"
#include "common/parameter.hpp"
#include <fstream>
#include "common/utils.hpp"
#include <cstdio>

namespace common {
class Visualization
{
public:
  Visualization()
  {
    {
      vins_result_path_ = common::Setting::getSingleton()->Get<std::string>("output_path");
      vins_result_file_name_ = common::Setting::getSingleton()->Get<std::string>("output_file_name");
      std::string file = vins_result_path_ + vins_result_file_name_;
      if (common::FileExists(file))
      {
        int result = std::system(("rm " + std::string(file)).c_str());
        if (result == 0)
          LOG(WARNING) << "Delete vins result exist file: " << file << " success";
      }
    }
    LOG(INFO) << "VINS result save path: " << vins_result_path_ + vins_result_file_name_;
    ros::NodeHandle n("~");
    pub_path_ = n.advertise<nav_msgs::Path>("path", 1000);
    pub_track_img_ = n.advertise<sensor_msgs::Image>("track_image", 1000);
    pub_point_cloud_ = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    path_.header.frame_id = "world";

    pub_camera_pose_visual_ = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    cam_pos_visual_ = std::make_shared<CameraPoseVisualization>(1, 0, 0, 1);
    cam_pos_visual_->setScale(0.4);
    cam_pos_visual_->setLineWidth(0.05);
  }

  void PublishPath(double timestamp, const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, int window_size);
  void PublishTrackImage(const cv::Mat &image, double t);
  void PublishPointCloud(double t);
  void PublishCameraPose(const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, Eigen::Vector3d *tic, Eigen::Matrix3d *ric,
                         int window_size, std_msgs::Header &header);

private:
  std::shared_ptr<CameraPoseVisualization> cam_pos_visual_ = nullptr;
  ros::Publisher pub_path_, pub_track_img_;
  ros::Publisher pub_point_cloud_;
  ros::Publisher pub_camera_pose_visual_;
  nav_msgs::Path path_;
  std::string vins_result_path_, vins_result_file_name_;
};
} // namespace common

#endif // VISUALIZATION_HPP
