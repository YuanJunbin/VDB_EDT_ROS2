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
 *     * Redistributions in binary form must reproduce the above copyright
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

#include "vdb_edt/vdbmap.h"
#include <cmath>
// NOTE(ROS2): Needed for tf2::doTransform on sensor_msgs::msg::PointCloud2
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_ros/create_timer_ros.h>

#include <unordered_map>
#include <algorithm>
#include <cstdint>

// # define ASYNC_SPINNER

VDBMap::VDBMap()
    : L_FREE(-0.13), L_OCCU(+1.01), L_THRESH(0.0), L_MIN(-2.0), L_MAX(+3.5), VOX_SIZE(0.2),
      START_RANGE(0.0), SENSOR_RANGE(5.0), HIT_THICKNESS(1), VERSION(1),
      MAX_UPDATE_DIST(20.0), EDT_UPDATE_DURATION(0.5), VIS_UPDATE_DURATION(10.0),
      VIS_MAP_MINX(-200.0), VIS_MAP_MINY(-200.0), VIS_MAP_MINZ(-1.0),
      VIS_MAP_MAXX(+200.0), VIS_MAP_MAXY(+200.0), VIS_MAP_MAXZ(10.0), VIS_SLICE_LEVEL(2.0),
      msg_ready_(false), occu_update_count_(0), dist_update_count_(0)
{
    node_handle_ = std::make_shared<rclcpp::Node>("vdb_map");
    node_name_ = get_node_name();

    setup_parameters();
    load_mapping_para();

    // LADY_AND_COW dataset transforms
    ref_transform_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    T_B_C_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    T_D_B_ << 0.971048, -0.120915, 0.206023, 0.00114049,
        0.157010, 0.973037, -0.168959, 0.04509360,
        -0.180038, 0.196415, 0.963850, 0.04307650,
        0.0, 0.0, 0.0, 1.0;

    // TF2 buffer/listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_handle_->get_clock());
    tf_buffer_->setCreateTimerInterface(std::make_shared<tf2_ros::CreateTimerROS>(node_handle_->get_node_base_interface(),
                                                                                  node_handle_->get_node_timers_interface()));
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_handle_, true);

    // Publishers
    occu_vis_pub_ = node_handle_->create_publisher<sensor_msgs::msg::PointCloud2>("/occ_grid", 5);
    slice_vis_pub_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>("/dist_slice", 5);

    // OpenVDB init & grids
    openvdb::initialize();
    grid_logocc_ = openvdb::FloatGrid::create(0.0);
    this->set_voxel_size(*grid_logocc_, VOX_SIZE);

    max_coor_dist_ = static_cast<int>(MAX_UPDATE_DIST / VOX_SIZE);
    max_coor_sqdist_ = max_coor_dist_ * max_coor_dist_;
    grid_distance_ = std::make_shared<DynamicVDBEDT>(max_coor_dist_);
    grid_distance_->initialize(dist_map_, VOX_SIZE, VERSION);
    grid_distance_->setAccessor(dist_map_);

    // Convert SENSOR_RANGE / START_RANGE from world (m) to index (voxels)
    {
        openvdb::Vec3d max_sense_dist(SENSOR_RANGE);
        openvdb::Vec3d sense_range_ijk = grid_logocc_->worldToIndex(max_sense_dist);
        SENSOR_RANGE = sense_range_ijk.x();

        openvdb::Vec3d min_sense_dist(START_RANGE);
        openvdb::Vec3d min_sense_ijk = grid_logocc_->worldToIndex(min_sense_dist);
        START_RANGE = min_sense_ijk.x();
    }

    // Timers
    update_edt_timer_ = node_handle_->create_wall_timer(
        std::chrono::duration<double>(EDT_UPDATE_DURATION),
        std::bind(&VDBMap::update_edtmap, this));
    update_vis_timer_ = node_handle_->create_wall_timer(
        std::chrono::duration<double>(VIS_UPDATE_DURATION),
        std::bind(&VDBMap::visualize_maps, this));

    // lady_and_cow dataset subscriptions
    lady_cow_cloud_sub_ = node_handle_->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/camera/depth_registered/points", rclcpp::SensorDataQoS(),
        std::bind(&VDBMap::lady_cow_cloud_callback, this, std::placeholders::_1));

    lady_cow_pose_sub_ = node_handle_->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/kinect/vrpn_client/estimated_transform", 10,
        std::bind(&VDBMap::lady_cow_pose_callback, this, std::placeholders::_1));

    // General synced dataset via message_filters + tf2 MessageFilter
    cloud_sub_mf_.subscribe(node_handle_.get(), pcl_topic, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    cloud_filter_ = std::make_unique<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
        cloud_sub_mf_, *tf_buffer_, worldframeId, 200, node_handle_);
    cloud_filter_->registerCallback(std::bind(&VDBMap::cloud_callback, this, std::placeholders::_1));
    RCLCPP_INFO(node_handle_->get_logger(), "[%s] VDBMap initialized.", node_name_.c_str());
}

VDBMap::~VDBMap()
{
    cloud_filter_.reset();
    tf_listener_.reset();
    tf_buffer_.reset();
}

//////////////////////////////////////////////////////////////////////////////////
// Node helpers

std::string VDBMap::get_node_name() const
{
    return node_handle_ ? node_handle_->get_name() : std::string("vdb_map");
}

rclcpp::Node::SharedPtr VDBMap::get_node_handle() const
{
    return node_handle_;
}

//////////////////////////////////////////////////////////////////////////////////
// Utilities

void VDBMap::set_voxel_size(openvdb::GridBase &grid, double vs)
{
    const openvdb::math::Vec3d offset(vs / 2., vs / 2., vs / 2.);
    openvdb::math::Transform::Ptr tf = openvdb::math::Transform::createLinearTransform(vs);
    tf->postTranslate(offset);
    grid.setTransform(tf);
}

void VDBMap::setup_parameters()
{
    // ROS2: declare parameters with defaults
    node_handle_->declare_parameter<std::string>("pcl_topic", "/robot_1/sensors/ouster/point_cloud");
    node_handle_->declare_parameter<std::string>("world_frame_id", "map");
    node_handle_->declare_parameter<std::string>("robot_frame_id", "base_link");
    node_handle_->declare_parameter<std::string>("data_set", "shimizu");

    node_handle_->declare_parameter<double>("l_free", L_FREE);
    node_handle_->declare_parameter<double>("l_occu", L_OCCU);
    node_handle_->declare_parameter<double>("l_max", L_MAX);
    node_handle_->declare_parameter<double>("l_min", L_MIN);
    node_handle_->declare_parameter<double>("l_thresh", L_THRESH);
    node_handle_->declare_parameter<double>("vox_size", VOX_SIZE);

    node_handle_->declare_parameter<double>("start_range", START_RANGE);
    node_handle_->declare_parameter<double>("sensor_range", SENSOR_RANGE);

    node_handle_->declare_parameter<int>("vdbedt_version", VERSION);
    node_handle_->declare_parameter<double>("max_update_dist", MAX_UPDATE_DIST);
    node_handle_->declare_parameter<double>("edt_update_duration", EDT_UPDATE_DURATION);

    node_handle_->declare_parameter<double>("vis_update_duration", VIS_UPDATE_DURATION);
    node_handle_->declare_parameter<double>("vis_slice_level", VIS_SLICE_LEVEL);
    node_handle_->declare_parameter<double>("vis_map_minx", VIS_MAP_MINX);
    node_handle_->declare_parameter<double>("vis_map_miny", VIS_MAP_MINY);
    node_handle_->declare_parameter<double>("vis_map_minz", VIS_MAP_MINZ);
    node_handle_->declare_parameter<double>("vis_map_maxx", VIS_MAP_MAXX);
    node_handle_->declare_parameter<double>("vis_map_maxy", VIS_MAP_MAXY);
    node_handle_->declare_parameter<double>("vis_map_maxz", VIS_MAP_MAXZ);
}

bool VDBMap::load_mapping_para()
{
    auto log = node_handle_->get_logger();

    if (node_handle_->get_parameter("pcl_topic", pcl_topic))
    {
        RCLCPP_INFO(log, "Parameter pcl_topic set to: %s", pcl_topic.c_str());
    }
    else
    {
        RCLCPP_ERROR(log, "Please set input point cloud topic before running the node.");
        return false;
    }

    if (node_handle_->get_parameter("world_frame_id", worldframeId))
    {
        RCLCPP_INFO(log, "Parameter world_frame_id set to: %s", worldframeId.c_str());
    }
    else
    {
        RCLCPP_ERROR(log, "Please set input world frame id before running the node.");
        return false;
    }

    if (node_handle_->get_parameter("robot_frame_id", robotframeId))
    {
        RCLCPP_INFO(log, "Parameter robot_frame_id set to: %s", robotframeId.c_str());
    }
    else
    {
        RCLCPP_ERROR(log, "Please set input robot frame id before running the node.");
        return false;
    }

    auto report_double = [&](const char *name, double &var)
    {
        if (node_handle_->get_parameter(name, var))
        {
            RCLCPP_INFO(log, "Overriding Parameter %s to: %f", name, var);
        }
        else
        {
            RCLCPP_WARN(log, "Using the default %s: %f", name, var);
        }
    };
    auto report_int = [&](const char *name, int &var)
    {
        if (node_handle_->get_parameter(name, var))
        {
            RCLCPP_INFO(log, "Overriding Parameter %s to: %d", name, var);
        }
        else
        {
            RCLCPP_WARN(log, "Using the default %s: %d", name, var);
        }
    };

    report_double("l_free", L_FREE);
    report_double("l_occu", L_OCCU);
    report_double("l_max", L_MAX);
    report_double("l_min", L_MIN);
    report_double("l_thresh", L_THRESH);
    report_double("vox_size", VOX_SIZE);
    report_double("start_range", START_RANGE);
    report_double("sensor_range", SENSOR_RANGE);
    report_int("vdbedt_version", VERSION);
    report_double("max_update_dist", MAX_UPDATE_DIST);
    report_double("edt_update_duration", EDT_UPDATE_DURATION);
    report_double("vis_update_duration", VIS_UPDATE_DURATION);
    report_double("vis_slice_level", VIS_SLICE_LEVEL);
    report_double("vis_map_minx", VIS_MAP_MINX);
    report_double("vis_map_miny", VIS_MAP_MINY);
    report_double("vis_map_minz", VIS_MAP_MINZ);
    report_double("vis_map_maxx", VIS_MAP_MAXX);
    report_double("vis_map_maxy", VIS_MAP_MAXY);
    report_double("vis_map_maxz", VIS_MAP_MAXZ);

    return true;
}

bool VDBMap::load_planning_para()
{
    std::cout << "planning parameters no set !" << std::endl;
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Callbacks

void VDBMap::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pc_msg)
{
    // TF2: MessageFilter ensures transform availability, but still catch exceptions
    geometry_msgs::msg::TransformStamped transform;
    try
    {
        transform = tf_buffer_->lookupTransform(
            worldframeId, pc_msg->header.frame_id, pc_msg->header.stamp, rclcpp::Duration::from_seconds(0.05));
    }
    catch (const tf2::TransformException &ex)
    {
        RCLCPP_ERROR(node_handle_->get_logger(), "TF lookup failed: %s", ex.what());
        return;
    }

    // Transform point cloud into world frame
    sensor_msgs::msg::PointCloud2 pc_world;
    try
    {
        tf2::doTransform(*pc_msg, pc_world, transform);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(node_handle_->get_logger(), "doTransform failed: %s", e.what());
        return;
    }

    // Extract origin from transform
    origin_ = Eigen::Vector3d(
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z);

    // Convert to PCL XYZ
    auto xyz = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(pc_world, *xyz);

    // update float occupancy map
    occu_update_count_++;
    std::cout << "Running " << occu_update_count_ << " updates." << std::endl;
    timing::Timer update_OCC_timer("UpdateOccu");
    this->update_occmap(grid_logocc_, origin_, xyz);
    update_OCC_timer.Stop();
    timing::Timing::Print(std::cout);
    msg_ready_ = true;
}

void VDBMap::lady_cow_pose_callback(const geometry_msgs::msg::TransformStamped::ConstSharedPtr &pose)
{
    WriteLock wlock(pose_queue_lock);
    pose_queue_.push(pose);
    while (static_cast<int>(pose_queue_.size()) > kPoseQueueSize)
    {
        pose_queue_.pop();
    }
}

void VDBMap::lady_cow_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud)
{
    {
        WriteLock wlock(pose_queue_lock);
        cloud_queue_.push(cloud);
    }
    sync_pose_and_cloud_fiesta();
}

bool VDBMap::sync_pose_and_cloud(geometry_msgs::msg::TransformStamped &latest_pose,
                                 const sensor_msgs::msg::PointCloud2 &latest_cloud)
{
    // Time tolerance window: 3ms (ROS2-safe arithmetic).
    const rclcpp::Duration tol = rclcpp::Duration::from_seconds(0.003);
#ifdef ASYNC_SPINNER
    std::this_thread::sleep_for(std::chrono::milliseconds(6));
#endif

    const rclcpp::Time cloud_time(latest_cloud.header.stamp);
    std::vector<geometry_msgs::msg::TransformStamped::ConstSharedPtr> candidates;

    {
        // We will pop inspected elements; operate under lock.
        WriteLock wlock(pose_queue_lock);
        size_t check_count = pose_queue_.size();

        while (check_count > 0)
        {
            auto pose_ptr = pose_queue_.front();
            const rclcpp::Time pose_time(pose_ptr->header.stamp);

            --check_count;
            if ((cloud_time - pose_time) > tol)
            {
                // Pose is too old relative to cloud; discard it.
                pose_queue_.pop();
                continue;
            }
            else if ((pose_time - cloud_time) < tol)
            {
                // Pose is a bit ahead, still within tolerance; keep scanning.
                // (No pop to retain for other clouds unless matched)
                // This branch mirrors the original logic; nothing to do.
            }
            else
            {
                // Pose within [cloud - tol, cloud + tol]: candidate.
                candidates.push_back(pose_ptr);
                pose_queue_.pop();
            }
        }
    }

    if (candidates.empty())
        return false;

    // Pick the closest in |t_pose - t_cloud|.
    long long best_cost = std::numeric_limits<long long>::max();
    size_t best_idx = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const auto dt = rclcpp::Time(candidates[i]->header.stamp) - cloud_time;
        const long long cost = std::llabs(dt.nanoseconds());
        if (cost < best_cost)
        {
            best_cost = cost;
            best_idx = i;
        }
    }
    if (best_idx >= candidates.size())
        return false;

    latest_pose = *candidates[best_idx];
    return true;
}

void VDBMap::sync_pose_and_cloud_fiesta()
{
    const rclcpp::Duration tol = rclcpp::Duration::from_seconds(0.003);

    while (true)
    {
        // Peek current cloud.
        sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_ptr;
        {
            WriteLock wlock(pose_queue_lock);
            if (cloud_queue_.empty())
                break;
            cloud_ptr = cloud_queue_.front();
        }

        bool has_new_pose = false;
        const rclcpp::Time cloud_time(cloud_ptr->header.stamp);

        {
            // Advance pose queue up to cloud_time + tol.
            WriteLock wlock(pose_queue_lock);
            while (pose_queue_.size() > 1 &&
                   rclcpp::Time(pose_queue_.front()->header.stamp) <= (cloud_time + tol))
            {
                latest_pose_ = *pose_queue_.front();
                pose_queue_.pop();
                has_new_pose = true;
            }

            // If no candidate pose beyond this point, we cannot match this cloud yet.
            if (pose_queue_.empty() ||
                rclcpp::Time(pose_queue_.front()->header.stamp) <= (cloud_time + tol))
            {
                break;
            }
        }

        if (!has_new_pose)
        {
            // No suitable pose for this cloud; drop it to keep up.
            WriteLock wlock(pose_queue_lock);
            cloud_queue_.pop();
            continue;
        }

        // Mark available for EDT update after occupancy update
        msg_ready_ = true;

        // ----- Build current transform from latest_pose_ -----
        ref_transform_ = cur_transform_;

        const Eigen::Vector3d translation(
            latest_pose_.transform.translation.x,
            latest_pose_.transform.translation.y,
            latest_pose_.transform.translation.z);
        const Eigen::Quaterniond rotation(
            latest_pose_.transform.rotation.w,
            latest_pose_.transform.rotation.x,
            latest_pose_.transform.rotation.y,
            latest_pose_.transform.rotation.z);

        cur_transform_.setIdentity();
        cur_transform_.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        cur_transform_.block<3, 1>(0, 3) = translation;
        cur_transform_ = cur_transform_ * T_D_B_ * T_B_C_;

        // Convert to TransformStamped for tf2::doTransform
        Eigen::Affine3d cur_eig_trans;
        cur_eig_trans.matrix() = cur_transform_;
        geometry_msgs::msg::TransformStamped cur_tf_msg = tf2::eigenToTransform(cur_eig_trans);
        cur_tf_msg.header.frame_id = worldframeId;
        cur_tf_msg.child_frame_id = robotframeId;
        cur_tf_msg.header.stamp = cloud_ptr->header.stamp;

        // Extract origin from transform
        origin_ = Eigen::Vector3d(
            cur_tf_msg.transform.translation.x,
            cur_tf_msg.transform.translation.y,
            cur_tf_msg.transform.translation.z);

        // ----- Cloud transform & occupancy update -----
        if (cloud_ptr->data.empty())
        {
            std::cout << "invalid point data!" << std::endl;
            WriteLock wlock(pose_queue_lock);
            cloud_queue_.pop();
            continue;
        }

        sensor_msgs::msg::PointCloud2 cloud_world;
        try
        {
            tf2::doTransform(*cloud_ptr, cloud_world, cur_tf_msg);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(node_handle_->get_logger(), "doTransform failed: %s", e.what());
            WriteLock wlock(pose_queue_lock);
            cloud_queue_.pop();
            continue;
        }

        auto xyz = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(cloud_world, *xyz);

        ++occu_update_count_;
        std::cout << "Running " << occu_update_count_ << " updates." << std::endl;
        timing::Timer update_OCC_timer("UpdateOccu");
        this->update_occmap(grid_logocc_, origin_, xyz);
        update_OCC_timer.Stop();
        timing::Timing::Print(std::cout);

        // Done with this cloud.
        WriteLock wlock(pose_queue_lock);
        cloud_queue_.pop();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Visualization

void VDBMap::visualize_maps()
{
    // visualize the occupancy map
    if (occu_vis_pub_->get_subscription_count() > 0)
    {
        sensor_msgs::msg::PointCloud2 cloud_vis;
        this->grid_message(grid_logocc_, cloud_vis);
        occu_vis_pub_->publish(cloud_vis);
    }

    // visualize the distance map
    if (slice_vis_pub_->get_subscription_count() > 0)
    {
        visualization_msgs::msg::Marker slice_maker;
        auto vis_coor = static_cast<int>(2.0 / VOX_SIZE);
        this->get_slice_marker(slice_maker, 100, VIS_SLICE_LEVEL, vis_coor * vis_coor);
        slice_vis_pub_->publish(slice_maker);
    }
}

void VDBMap::grid_to_pcl(openvdb::FloatGrid::ConstPtr grid,
                         openvdb::FloatGrid::ValueType thresh,
                         std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> &pc_out)
{
    pcl::PointXYZI point_xyzi;

    using value_type = openvdb::FloatGrid::ValueType;
    using itr_type = openvdb::FloatGrid::ValueOnCIter;

    const openvdb::math::Transform &grid_tf(grid->transform());

    for (itr_type itr = grid->cbeginValueOn(); itr.test(); ++itr)
    {
        if (!itr.isVoxelValue())
            continue;
        value_type val = itr.getValue();
        if (val < thresh)
            continue;

        openvdb::Coord ijk = itr.getCoord();
        openvdb::Vec3d p = grid_tf.indexToWorld(ijk);

        point_xyzi.x = static_cast<float>(p.x());
        point_xyzi.y = static_cast<float>(p.y());
        point_xyzi.z = static_cast<float>(p.z());
        point_xyzi.intensity = static_cast<float>(val);
        pc_out->points.push_back(point_xyzi);
    }
    pc_out->width = static_cast<uint32_t>(pc_out->points.size());
    pc_out->height = 1;
}

void VDBMap::grid_message(const openvdb::FloatGrid::Ptr &grid,
                          sensor_msgs::msg::PointCloud2 &disp_msg)
{
    auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    this->grid_to_pcl(grid, static_cast<openvdb::FloatGrid::ValueType>(L_THRESH), pcl_cloud);
    pcl::toROSMsg(*pcl_cloud, disp_msg);
    disp_msg.header.frame_id = worldframeId;
    disp_msg.header.stamp = node_handle_->now();
}

////////////////////////////////////////////////////////////////////////////////
// EDT

void VDBMap::update_edtmap()
{
    if (!msg_ready_)
    {
        return;
    }

    // Lock
    std::unique_lock<std::shared_mutex> lk(map_mutex);

    // update float occupancy map
    dist_update_count_++;
    std::cout << "Running " << dist_update_count_ << " updates." << std::endl;
    timing::Timer update_DIST_timer("UpdateDIST");
    this->grid_distance_->update();
    update_DIST_timer.Stop();
    timing::Timing::Print(std::cout);
    std::cout << "changed: " << grid_distance_->sum_occ_changed
              << " raised: " << grid_distance_->sum_raised_num
              << " lowered: " << grid_distance_->sum_lowered_num << std::endl;
    msg_ready_ = false;
}

////////////////////////////////////////////////////////////////////////////////
// Occupancy update (ray casting)

struct CoordHash
{
    std::size_t operator()(const openvdb::Coord &c) const noexcept
    {
        std::uint64_t h = 0xCBF29CE484222325ull;
        auto mix = [&](int v)
        {
            std::uint64_t x = static_cast<std::uint32_t>(v);
            h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        };
        mix(c.x());
        mix(c.y());
        mix(c.z());
        return static_cast<std::size_t>(h);
    }
};

void VDBMap::update_occmap(openvdb::FloatGrid::Ptr grid_map,
                           const Eigen::Vector3d &origin,
                           std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> xyz)
{
    // Accumulate log-odds deltas per voxel (no grid access here).
    std::unordered_map<openvdb::Coord, float, CoordHash> ll_delta;
    ll_delta.reserve(xyz->size() * 8);

    const openvdb::Vec3d origin_ijk =
        grid_map->worldToIndex(openvdb::Vec3d(origin.x(), origin.y(), origin.z()));

    for (const pcl::PointXYZ &pt : *xyz)
    {
        const openvdb::Vec3d p_ijk =
            grid_map->worldToIndex(openvdb::Vec3d(pt.x, pt.y, pt.z));
        openvdb::Vec3d dir = p_ijk - origin_ijk;
        const double range = dir.length();
        if (range <= 1e-6)
            continue;
        dir.normalize();

        openvdb::math::Ray<double> ray(origin_ijk, dir);
        openvdb::math::DDA<openvdb::math::Ray<double>, 0> dda(
            ray, 0.0, std::min(SENSOR_RANGE, range));

        // Free space along the ray (exclude the first hit voxel).
        for (;;)
        {
            const openvdb::Coord cur = dda.voxel();
            // If stepping further would exceed/meet maxTime, `cur` is the hit voxel → don't add FREE.
            if (!(dda.time() < dda.maxTime()))
                break;
            ll_delta[cur] += static_cast<float>(L_FREE);
            dda.step();
        }

        // Hit (with thickness) only if not truncated by sensor range.
        if (range <= SENSOR_RANGE)
        {
            // first hit voxel
            {
                const openvdb::Coord hit = dda.voxel();
                ll_delta[hit] += static_cast<float>(L_OCCU);
            }
            // extra thickness: step then add, until maxTime
            for (int i = 1; i < HIT_THICKNESS; ++i)
            {
                if (!(dda.time() < dda.maxTime()))
                    break;
                dda.step();
                ll_delta[dda.voxel()] += static_cast<float>(L_OCCU);
            }
        }
    }

    // Batch apply under lock: write occupancy & notify EDT
    {
        std::unique_lock<std::shared_mutex> lk(map_mutex);

        auto acc = grid_map->getAccessor();
        const float occ_thresh = static_cast<float>(L_THRESH);

        for (const auto &kv : ll_delta)
        {
            const openvdb::Coord &ijk = kv.first;
            const float dL = kv.second;

            float ll_old = 0.0f;
            const bool known = acc.probeValue(ijk, ll_old);

            const float ll_new = std::clamp(ll_old + dL,
                                            static_cast<float>(L_MIN),
                                            static_cast<float>(L_MAX));

            if (!known)
            {
                // Unknown → write & init EDT cell
                grid_distance_->dist_acc_->setValueOn(ijk);
                acc.setValueOn(ijk, ll_new);
                if (ll_new >= occ_thresh)
                {
                    openvdb::Coord c = ijk; // unknown -> occupied
                    grid_distance_->setObstacle(c);
                }
                continue;
            }

            const bool wasOcc = (ll_old >= occ_thresh);
            const bool nowOcc = (ll_new >= occ_thresh);

            acc.setValueOn(ijk, ll_new);

            if (!wasOcc && nowOcc)
            {
                openvdb::Coord c = ijk; // free -> occupied
                grid_distance_->setObstacle(c);
            }
            else if (wasOcc && !nowOcc)
            {
                openvdb::Coord c = ijk; // occupied -> free
                grid_distance_->removeObstacle(c);
            }
        }
    }
}

// void VDBMap::update_occmap(openvdb::FloatGrid::Ptr grid_map,
//                            const Eigen::Vector3d &origin,
//                            std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> xyz)
// {
//     std::unique_lock<std::shared_mutex> lk(map_mutex);
//     auto grid_acc = grid_map->getAccessor();

//     openvdb::Vec3d origin3d(origin.x(), origin.y(), origin.z());
//     openvdb::Vec3d origin_ijk = grid_map->worldToIndex(origin3d);

//     for (auto point = xyz->begin(); point != xyz->end(); ++point)
//     {
//         openvdb::Vec3d p_xyz(point->x, point->y, point->z);
//         openvdb::Vec3d p_ijk = grid_map->worldToIndex(p_xyz);
//         openvdb::Vec3d dir(p_ijk - origin_ijk);
//         double range = dir.length();
//         dir.normalize();

//         // Note: real sensor range should strictly be larger than sensor_range
//         bool truncated = false;
//         openvdb::math::Ray<double> ray(origin_ijk, dir);
//         openvdb::math::DDA<openvdb::math::Ray<double>, 0> dda(ray, 0.0, std::min(SENSOR_RANGE, range));

//         // decrease occupancy
//         do
//         {
//             openvdb::Coord ijk(dda.voxel());

//             float ll_old = 0.0f;
//             bool isKnown = grid_acc.probeValue(ijk, ll_old);
//             float ll_new = std::max(static_cast<float>(L_MIN), ll_old + static_cast<float>(L_FREE));

//             if (!isKnown)
//             {
//                 grid_distance_->dist_acc_->setValueOn(ijk);
//             } // unknown -> free -> EDT initialize
//             else if (ll_old >= 0.0f && ll_new < 0.0f)
//             {
//                 grid_distance_->removeObstacle(ijk);
//             } // occupied -> free -> EDT RemoveObstacle

//             grid_acc.setValueOn(ijk, ll_new);
//             dda.step();

//         } while (dda.time() < dda.maxTime());

//         // increase occupancy
//         if ((!truncated) && (range <= SENSOR_RANGE))
//         {
//             for (int i = 0; i < HIT_THICKNESS; ++i)
//             {
//                 openvdb::Coord ijk(dda.voxel());

//                 float ll_old = 0.0f;
//                 bool isKnown = grid_acc.probeValue(ijk, ll_old);
//                 float ll_new = std::min(static_cast<float>(L_MAX), ll_old + static_cast<float>(L_OCCU));

//                 if (!isKnown)
//                 {
//                     grid_distance_->dist_acc_->setValueOn(ijk);
//                 } // unknown -> occupied -> EDT SetObstacle
//                 else if (ll_old < 0.0f && ll_new >= 0.0f)
//                 {
//                     grid_distance_->setObstacle(ijk);
//                 } // free -> occupied -> EDT SetObstacle

//                 grid_acc.setValueOn(ijk, ll_new);
//                 dda.step();
//             }
//         } // process obstacle
//     } // end inserting
// }

////////////////////////////////////////////////////////////////////////////////
// Slice visualization helpers

void VDBMap::get_slice_marker(visualization_msgs::msg::Marker &marker, int marker_id, double slice, double max_sqdist)
{
    marker.header.frame_id = worldframeId;
    marker.header.stamp = node_handle_->now();
    marker.id = marker_id;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    marker.scale.x = VOX_SIZE;
    marker.scale.y = VOX_SIZE;
    marker.scale.z = VOX_SIZE;
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;

    marker.points.clear();
    marker.colors.clear();

    // iterate the map
    std_msgs::msg::ColorRGBA c;
    openvdb::Vec3d map_min(VIS_MAP_MINX, VIS_MAP_MINY, VIS_MAP_MINZ);
    openvdb::Vec3d map_max(VIS_MAP_MAXX, VIS_MAP_MAXY, VIS_MAP_MAXZ);
    auto coor_min = dist_map_->worldToIndex(map_min);
    auto coor_max = dist_map_->worldToIndex(map_max);

    const int slice_idx = static_cast<int>(std::round(dist_map_->worldToIndex(openvdb::Vec3d(0.0, 0.0, slice)).z()));
    if (slice_idx < coor_min.z() || slice_idx > coor_max.z())
    // if (slice < coor_min.z() || slice > coor_max.z())
    {
        std::cout << "slice number is out of boundary!" << std::endl;
        return;
    }

    // int z = dist_map_->worldToIndex(openvdb::Vec3d(slice)).x();
    int z = slice_idx;
    std::shared_lock<std::shared_mutex> lk(map_mutex);
    for (int x = coor_min.x(); x <= coor_max.x(); ++x)
    {
        for (int y = coor_min.y(); y <= coor_max.y(); ++y)
        {
            openvdb::math::Coord target_cell(x, y, z);
            openvdb::math::Coord nearest_obst;
            auto cell_dist = grid_distance_->query_sq_distance(target_cell, nearest_obst);

            if (cell_dist < 0 || cell_dist >= max_coor_sqdist_)
            { // unknown cells
                continue;
            }
            auto world_xyz = dist_map_->indexToWorld(target_cell);

            geometry_msgs::msg::Point p;
            p.x = world_xyz.x();
            p.y = world_xyz.y();
            p.z = world_xyz.z();

            double norm = std::sqrt(static_cast<double>(cell_dist));
            double denom = std::sqrt(max_sqdist);
            c = rainbow_color_map((denom > 0.0) ? std::min(1.0, norm / denom) : 1.0);

            marker.points.push_back(p);
            marker.colors.push_back(c);

        } // end y loop
    } // end x loop
}

std_msgs::msg::ColorRGBA VDBMap::rainbow_color_map(double h)
{
    std_msgs::msg::ColorRGBA color;
    color.a = 1;
    // blend over HSV-values (more colors)

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = static_cast<int>(floor(h));
    f = h - i;
    if (!(i & 1))
        f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i)
    {
    case 6:
    case 0:
        color.r = v;
        color.g = n;
        color.b = m;
        break;
    case 1:
        color.r = n;
        color.g = v;
        color.b = m;
        break;
    case 2:
        color.r = m;
        color.g = v;
        color.b = n;
        break;
    case 3:
        color.r = m;
        color.g = n;
        color.b = v;
        break;
    case 4:
        color.r = n;
        color.g = m;
        color.b = v;
        break;
    case 5:
        color.r = v;
        color.g = m;
        color.b = n;
        break;
    default:
        color.r = 1;
        color.g = 0.5;
        color.b = 0.5;
        break;
    }
    return color;
}
