cmake_minimum_required(VERSION 3.0.2)
project(vins)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  rosbag
  camera_models
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(test_feature_track test/test_feature_tracker.cpp)
target_link_libraries(test_feature_track glog gflags)