//
// Created by weihao on 23-9-28.
//
#include "visualization.hpp"

namespace common {

void Visualization::PublishPath(const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, int WINDOW_SIZE)
{
  Eigen::Quaterniond tmp_Q = Eigen::Quaterniond(Rs[WINDOW_SIZE]);
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = Ps[WINDOW_SIZE].x();
  pose.pose.position.y = Ps[WINDOW_SIZE].y();
  pose.pose.position.z = Ps[WINDOW_SIZE].z();
  pose.pose.orientation.x = tmp_Q.x();
  pose.pose.orientation.y = tmp_Q.y();
  pose.pose.orientation.z = tmp_Q.z();
  pose.pose.orientation.w = tmp_Q.w();

  path_.poses.push_back(pose);
  pub_path_.publish(path_);
}
void Visualization::PublishTrackImage(const cv::Mat &image, double t)
{
  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time(t);
  sensor_msgs::ImagePtr img_track_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  pub_track_img_.publish(img_track_msg);
}
void Visualization::PublishPointCloud(double t) {

}
} // namespace common
