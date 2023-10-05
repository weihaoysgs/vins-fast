find_package(catkin REQUIRED COMPONENTS
roscpp
std_msgs
rosbag
geometry_msgs
nav_msgs
tf
cv_bridge
camera_models
image_transport)
find_package(OpenCV 3 REQUIRED)
find_package(Ceres 1.14.0 REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/eigen
  ${PROJECT_SOURCE_DIR}/thirdparty/sophus
  ${PROJECT_SOURCE_DIR}
)

