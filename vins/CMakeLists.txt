cmake_minimum_required(VERSION 3.0.2)
project(vins)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb ${CMAKE_CXX_FLAGS} -lpthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")

include(cmake/packages.cmake)
add_library(${PROJECT_NAME} SHARED
estimator/feature_tracker.hpp
estimator/feature_tracker.cpp
estimator/estimator.cpp
estimator/estimator.hpp
estimator/id_features.hpp
estimator/frame_feature.hpp
estimator/feature_manager.cpp
estimator/feature_manager.hpp
common/parameter.hpp
common/parameter.cpp
common/utils.hpp
common/time_calculate.hpp
common/visualization.hpp
common/visualization.cpp
common/couttool.hpp
)

target_link_libraries(${PROJECT_NAME} glog gflags ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(test_feature_track test/test_feature_tracker.cpp)
target_link_libraries(test_feature_track ${PROJECT_NAME})