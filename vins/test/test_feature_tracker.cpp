#include <glog/logging.h>
#include <gflags/gflags.h>
#include "estimator/estimator.hpp"
#include <thread>
#include <rosbag/view.h>
#include <rosbag/bag.h>
#include "common/utils.hpp"
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>

DEFINE_string(config_file_path, "/home/weihao/codespace/whvio_ws/src/vins-fast/vins/config/euroc/euroc_stero.yaml", "vins yaml config file path");

void GoWithRosBag(estimator::Estimator &vins_estimator)
{
  const std::string rosbag_path = common::Setting::getSingleton()->Get<std::string>("ros_bag_path");
  LOG_ASSERT(common::FileExists(rosbag_path));
  rosbag::Bag bag(rosbag_path);

  LOG(INFO) << "Run in " << rosbag_path;
  LOG_ASSERT(bag.isOpen()) << "Bag file open failed";

  std::shared_ptr<rosbag::View> bag_view = std::make_shared<rosbag::View>(bag);
  for (const rosbag::MessageInstance &m : (*bag_view))
  {
    if (m.getTopic() == vins_estimator.IMU_TOPIC_NAME && vins_estimator.USE_IMU)
    {
      sensor_msgs::Imu::ConstPtr msg = m.template instantiate<sensor_msgs::Imu>();
      /// IMU callback fun
    }
    if (m.getTopic() == vins_estimator.IMG0_TOPIC_NAME)
    {
      sensor_msgs::Image::ConstPtr msg = m.template instantiate<sensor_msgs::Image>();
      vins_estimator.Image0Callback(msg);
    }
    if (m.getTopic() == vins_estimator.IMG1_TOPIC_NAME)
    {
      sensor_msgs::Image::ConstPtr msg = m.template instantiate<sensor_msgs::Image>();
      vins_estimator.Image1Callback(msg);
    }
    usleep(1000);
  }
  bag.close();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vins_estimator_");

  google::InitGoogleLogging("TestFeatureTracking");
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags(&argc, &argv, true);

  common::Setting::getSingleton()->InitParamSetting(fLS::FLAGS_config_file_path);

  estimator::Estimator vins;
  std::thread go(GoWithRosBag, std::ref(vins));

  ros::spin();
}