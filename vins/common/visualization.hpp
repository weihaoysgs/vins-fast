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

namespace common
{
class Visualization
{
public:
  Visualization()
  {
    ros::NodeHandle n("~");
    pub_path_ = n.advertise<nav_msgs::Path>("path", 1000);
    pub_track_img_ = n.advertise<sensor_msgs::Image>("track_image", 1000);
    pub_point_cloud_ = n.advertise<sensor_msgs::PointCloud>("track_image", 1000);
    path_.header.frame_id = "world";
  }

  void PublishPath(const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, int window_size);
  void PublishTrackImage(const cv::Mat &image, double t);
  void PublishPointCloud(double t);
private:
  ros::Publisher pub_path_, pub_track_img_;
  ros::Publisher pub_point_cloud_;
  nav_msgs::Path path_;
};
}

#endif // VISUALIZATION_HPP
