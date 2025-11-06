/*
 * Copyright (c) Deron (Delong Zhu)
   The Chinese University of Hong Kong
   Carnegie Mellon University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions of binary form must reproduce the above copyright
 *       notice, this list of conditions and the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _VDBMAP_H_
#define _VDBMAP_H_

#include <queue>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <limits>

// C++17 locks (replace boost::shared_mutex)
#include <shared_mutex>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/timer.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/message_filter.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// NOTE(ROS2): doTransform for PointCloud2 lives in tf2_sensor_msgs (used in .cpp).
// #include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

// message_filters in ROS2
#include <message_filters/subscriber.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>

#include "vdb_edt/dynamicVDBEDT.h"
#include "vdb_edt/timing.h"

// Prefer constexpr over macro for compile-time constants.
static constexpr int kPoseQueueSize = 20;

class VDBMap{

public:
    VDBMap();
    ~VDBMap();

private:
    // General parameters
    std::string pcl_topic;
    std::string worldframeId;
    std::string robotframeId;
    std::string dataset;

    // Mapping parameters
    double L_FREE, L_OCCU, L_THRESH, L_MIN, L_MAX, VOX_SIZE;
    double START_RANGE, SENSOR_RANGE;
    int HIT_THICKNESS;

    // VDB map
    int VERSION;
    double MAX_UPDATE_DIST;
    double VIS_MAP_MINX, VIS_MAP_MINY, VIS_MAP_MINZ; // for visualization
    double VIS_MAP_MAXX, VIS_MAP_MAXY, VIS_MAP_MAXZ; // for visualization
    double EDT_UPDATE_DURATION;
    double VIS_UPDATE_DURATION;
    double VIS_SLICE_LEVEL; // in meters


    std::string node_name_;
    // ROS2: keep a single rclcpp node. No private node handle is needed.
    rclcpp::Node::SharedPtr node_handle_;

public:
    std::string get_node_name() const;
    rclcpp::Node::SharedPtr get_node_handle() const;

    // read-write lock (C++17)
    using Lock = std::shared_mutex;
    using WriteLock = std::unique_lock<Lock>;
    using ReadLock = std::shared_lock<Lock>;
    Lock pose_queue_lock;

    // common tool functions
    void setup_parameters();
    bool load_mapping_para();
    bool load_planning_para();

    // for visulization
    rclcpp::TimerBase::SharedPtr update_vis_timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occu_vis_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr slice_vis_pub_;
    // ROS2: rclcpp::Timer uses callback without TimerEvent by default.
    void visualize_maps();

    // general dataset with tf and point cloud
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_mf_;
    std::unique_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> cloud_filter_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pc_msg);

    /*** specially designed for lady_and_cow dataset
         there is a sync problem in this dataset
    */
    bool msg_ready_;
    // ROS2: store numeric origin as Eigen::Vector3d.
    Eigen::Vector3d origin_;
    geometry_msgs::msg::TransformStamped latest_pose_;

    // NOTE(ROS2): dataset-specific subscriptions
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lady_cow_cloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr lady_cow_pose_sub_;

    
    // Use ConstSharedPtr to avoid large copies and reduce memory footprint.
    std::queue<geometry_msgs::msg::TransformStamped::ConstSharedPtr> pose_queue_;
    std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_queue_;

public:
    void sync_pose_and_cloud_fiesta();
    bool sync_pose_and_cloud(geometry_msgs::msg::TransformStamped &latest_pose,
                             const sensor_msgs::msg::PointCloud2 &latest_cloud);
    void lady_cow_pose_callback(const geometry_msgs::msg::TransformStamped::ConstSharedPtr& pose);
    void lady_cow_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud);

private: // occupancy map
    std::shared_mutex map_mutex;

    // occupancy map
    openvdb::FloatGrid::Ptr grid_logocc_;

    // major functions
    void set_voxel_size(openvdb::GridBase& grid, double vs);
    void update_occmap(openvdb::FloatGrid::Ptr grid_map,
                       const Eigen::Vector3d& origin,
                       std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> xyz);

    // visualization
    void grid_to_pcl(openvdb::FloatGrid::ConstPtr grid,
                     openvdb::FloatGrid::ValueType thresh,
                     std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>>& pc_out);
    void grid_message(const openvdb::FloatGrid::Ptr& grid,
                      sensor_msgs::msg::PointCloud2 &disp_msg);

private: // distance map

    using CoordList = std::vector<openvdb::math::Coord>;

    // distance map
    int max_coor_dist_;
    int max_coor_sqdist_;
    EDTGrid::Ptr dist_map_;
    std::shared_ptr<DynamicVDBEDT> grid_distance_;

    // functions for updating distance map
    rclcpp::TimerBase::SharedPtr update_edt_timer_;
    void update_edtmap(); // NOTE(ROS2): timer callback without event object.

    std_msgs::msg::ColorRGBA rainbow_color_map(double h);
    void get_slice_marker(visualization_msgs::msg::Marker &marker, int marker_id,
                          double slice, double max_sqdist);

private: // pose correction for lady and cow dataset
    int occu_update_count_;
    int dist_update_count_;
    Eigen::Matrix4d cur_transform_;
    Eigen::Matrix4d ref_transform_;
    Eigen::Matrix4d T_B_C_, T_D_B_;
};

#endif
