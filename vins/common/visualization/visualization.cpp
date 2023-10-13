//
// Created by weihao on 23-9-28.
//
#include "visualization.hpp"

namespace common {

void Visualization::PublishPath(double timestamp, const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, int WINDOW_SIZE)
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
  {
    /// write result to file
    std::string VINS_RESULT_PATH = vins_result_path_ + vins_result_file_name_;
    std::ofstream f_out(VINS_RESULT_PATH, std::ios::app);
    f_out.setf(std::ios::fixed, std::ios::floatfield);
    f_out.precision(9);
    f_out << timestamp << " ";
    f_out.precision(5);
    f_out << Ps[WINDOW_SIZE].x() << " " << Ps[WINDOW_SIZE].y() << " " << Ps[WINDOW_SIZE].z() << " " << tmp_Q.w() << " "
          << tmp_Q.x() << " " << tmp_Q.y() << " " << tmp_Q.z() << std::endl;
    f_out.close();
  }
}
void Visualization::PublishTrackImage(const cv::Mat &image, double t)
{
  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time(t);
  sensor_msgs::ImagePtr img_track_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  pub_track_img_.publish(img_track_msg);
}

void Visualization::PublishPointCloud(double t) { }

void Visualization::PublishCameraPose(const Eigen::Vector3d *Ps, const Eigen::Matrix3d *Rs, Eigen::Vector3d *tic,
                                      Eigen::Matrix3d *ric, int window_size, std_msgs::Header &header)
{
  int i = window_size - 1;
  Eigen::Vector3d P = Ps[i] + Rs[i] * tic[0];
  Eigen::Quaterniond R = Eigen::Quaterniond(Rs[i] * ric[0]);
  cam_pos_visual_->Reset();
  cam_pos_visual_->AddPose(P, R);
  cam_pos_visual_->publish_by(pub_camera_pose_visual_, header);
}
} // namespace common
