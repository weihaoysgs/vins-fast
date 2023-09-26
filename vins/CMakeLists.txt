cmake_minimum_required(VERSION 3.0.2)
project(vins)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  camera_models
)

catkin_package(

)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
