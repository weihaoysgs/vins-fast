cmake_minimum_required(VERSION 3.0.2)
project(vins)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb ${CMAKE_CXX_FLAGS} -lpthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")

include(cmake/packages.cmake)
add_library(${PROJECT_NAME} SHARED
estimator/feature_tracker.cpp
estimator/estimator.cpp
estimator/feature_manager.cpp
common/parameter.cpp
common/visualization/cam_pose_visualization.cpp
common/visualization/visualization.cpp
factor/pose_local_parameterization.cpp
factor/projection2frame1camera_factor.cpp
factor/projection1frame2camera_factor.cpp
factor/projection2frame2camera_factor.cpp
factor/marginalization/marg_factor.cpp
factor/marginalization/marg_info.cpp
factor/marginalization/marg_residual_block_info.cpp
factor/imu/imu_factor.cpp
factor/imu/imu_integration_base.cpp
initial/initial_alignment.cpp
)

target_link_libraries(${PROJECT_NAME} glog gflags ${OpenCV_LIBS} ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_executable(test_feature_track test/test_feature_tracker.cpp)
target_link_libraries(test_feature_track ${PROJECT_NAME})

add_executable(test_imu_integration test/test_imu_integration.cpp)
target_link_libraries(test_imu_integration ${PROJECT_NAME})

add_executable(test_ins_system test/ins/test_imu_factor.cpp)
target_link_libraries(test_ins_system ${PROJECT_NAME})