cmake_minimum_required(VERSION 3.10)
project(vdb_edt)

# C++17 & basic warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -O3)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules" ${CMAKE_MODULE_PATH})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenVDB REQUIRED)

find_package(Boost REQUIRED)

# Include dirs
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_library(vdb_edt STATIC
  src/dynamicVDBEDT.cpp
  src/timing.cc
)
target_include_directories(vdb_edt PUBLIC include)
target_link_libraries(vdb_edt PUBLIC OpenVDB::openvdb)
target_compile_features(vdb_edt PUBLIC cxx_std_17)

add_executable(vdb_edt_node
  src/vdb_edt_node.cpp
  src/vdbmap.cpp
)
target_link_libraries(vdb_edt_node
  vdb_edt
)
ament_target_dependencies(vdb_edt_node
  rclcpp
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  message_filters
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  pcl_conversions
)

# Install
install(TARGETS vdb_edt vdb_edt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
        DESTINATION share/${PROJECT_NAME})
ament_package()
