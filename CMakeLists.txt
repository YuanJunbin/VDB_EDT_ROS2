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
  src/vdbmap.cpp
  src/frontier_cluster.cpp
)
# target_include_directories(vdb_edt PUBLIC include)
target_include_directories(vdb_edt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vdb_edt PUBLIC OpenVDB::openvdb)

ament_target_dependencies(vdb_edt PUBLIC
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
  Eigen3
)

add_executable(vdb_edt_node
  src/vdb_edt_node.cpp
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

# Install library
install(TARGETS vdb_edt
  EXPORT export_vdb_edt
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install executable, cannot install with library, otherwise adding exec as depend will cause build error
install(TARGETS vdb_edt_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING 
    PATTERN "*.h"
    PATTERN "*.hxx"
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})

install(EXPORT export_vdb_edt
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_dependencies(OpenVDB Eigen3)
ament_export_targets(export_vdb_edt HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(vdb_edt)
ament_package()
