#include "vdb_edt/frontier_cluster.h"

#include <queue>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>

FrontierManager::FrontierManager()
    : min_cluster_size_(10),          // 一个 cluster 里至少 10 个体素
      down_sample_rate_(2.0),         // 视点抽样：voxel_size * 2 做 VoxelGrid
      cluster_size_xy_(3.0),          // XY 上超过 3m 就考虑 split

      safe_robot_r_(0.5),             // EDT 安全距离
      candidate_dphi_(M_PI / 8.0),    // 每 22.5 度采样一个 viewpoint
      viewpoint_rmin_(0.5),           // viewpoint 离 frontier 中心最小半径
      viewpoint_rmax_(2.0),           // 最大半径
      candidate_rnum_(3),             // 半径上采 3 个圈
      min_visib_num_(3),              // 至少看到 3 个 downsample 后 cell 才算有效视点

      expand_x_(1.0),                 // 更新 box 周围再扩 5m 做 clustering
      expand_y_(1.0),
      expand_z_(1.0)
{

}

void FrontierManager::initialize(openvdb::BoolGrid::Ptr& external_grid)
{
    grid_clustered_frontier_ = openvdb::BoolGrid::create(false);
    first_new_cluster_ = frontiers_.end();

    grid_clustered_frontier_->setTransform(external_grid->transform().copy());
    const double voxel_size = external_grid->voxelSize()[0];
    const double safe_index_dist = safe_robot_r_ / voxel_size;
    safe_sq_dist_ = safe_index_dist * safe_index_dist;
}

openvdb::CoordBBox FrontierManager::expand_update_box(const openvdb::CoordBBox &update_box,
                                                      double voxel_size)
{
    openvdb::CoordBBox box = update_box;
    const int nx = static_cast<int>(std::ceil(expand_x_ / voxel_size));
    const int ny = static_cast<int>(std::ceil(expand_y_ / voxel_size));
    const int nz = static_cast<int>(std::ceil(expand_z_ / voxel_size));

    box.expand(openvdb::Coord(nx, ny, nz));
    return box;
}

bool FrontierManager::cluster_overlap_box(const FrontierCluster &cluster,
                                          const openvdb::CoordBBox &expanded_box)
{
    return expanded_box.hasOverlap(cluster.coord_box_);
}

bool FrontierManager::any_old_frontier_off(const FrontierCluster &cluster,
                                           const openvdb::BoolGrid::ConstAccessor &frontier_acc) // latest full frontier grid
{
    for (const auto &coord : cluster.cell_coords_)
    {
        if (!frontier_acc.isValueOn(coord))
        {
            return true;
        }
    }
    return false;
}

void FrontierManager::unflag_changed_cluster(const FrontierCluster &cluster)
{
    auto grid_acc = grid_clustered_frontier_->getAccessor();
    for (const auto &ijk : cluster.cell_coords_)
    {
        grid_acc.setValueOff(ijk, false);
    }
}

void FrontierManager::reset_changed_clusters(const openvdb::CoordBBox &expanded_box,
                                             const openvdb::BoolGrid::ConstAccessor &frontier_acc)
{
    removed_ids_.clear();
    int rmv_idx = 0;
    for (auto it = frontiers_.begin(); it != frontiers_.end(); /* no ++ */)
    {
        if (cluster_overlap_box(*it, expanded_box) &&
            any_old_frontier_off(*it, frontier_acc))
        {
            unflag_changed_cluster(*it);

            // it->cells_.clear(); it->cells_.shrink_to_fit();
            // it->cell_coords_.clear(); it->cell_coords_.shrink_to_fit();
            // it->filtered_cells_.clear(); it->filtered_cells_.shrink_to_fit();
            // it->paths_.clear(); it->costs_.clear();

            it = frontiers_.erase(it);
            removed_ids_.push_back(rmv_idx); // position index
        }
        else
        {
            ++rmv_idx;
            ++it;
        }
    }

    for (auto it = dormant_frontiers_.begin(); it != dormant_frontiers_.end(); /* no ++ */)
    {
        if (cluster_overlap_box(*it, expanded_box) &&
            any_old_frontier_off(*it, frontier_acc))
        {
            unflag_changed_cluster(*it);
            it = dormant_frontiers_.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return;
}

void FrontierManager::frontier_clustering(const openvdb::BoolGrid::Ptr &grid_frontier,
                                          const openvdb::CoordBBox &expanded_box)
{
    openvdb::BoolGrid::ConstAccessor frontier_acc = grid_frontier->getConstAccessor();
    openvdb::BoolGrid::Accessor cluster_acc = grid_clustered_frontier_->getAccessor();

    // All frontiers in box
    for (auto iter = grid_frontier->cbeginValueOn(); iter; ++iter)
    {
        if (!iter.isVoxelValue() || !expanded_box.isInside(iter.getCoord()))
        {
            continue;
        }

        const openvdb::Coord &ijk = iter.getCoord();
        // unclustered
        if (!cluster_acc.isValueOn(ijk))
        {
            expand_frontier(ijk, frontier_acc, cluster_acc);
        }
    }

    split_large_frontiers(tmp_frontiers_);
}

void FrontierManager::expand_frontier(const openvdb::Coord &seed,
                                      openvdb::BoolGrid::ConstAccessor &frontier_acc,
                                      openvdb::BoolGrid::Accessor &cluster_acc)
{
    static const int d26[26][3] =
        {{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, 0, -1}, {0, 0, 1}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1}, {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1}, {1, 1, -1}, {1, 1, 0}, {1, 1, 1}};

    std::queue<openvdb::Coord> expand_queue;
    std::vector<openvdb::Coord> coords;
    std::vector<Eigen::Vector3d> points;

    expand_queue.push(seed);
    coords.push_back(seed);
    cluster_acc.setValueOn(seed);

    const auto &tf = grid_clustered_frontier_->transform();

    {
        const auto pw = tf.indexToWorld(seed);
        points.emplace_back(pw.x(), pw.y(), pw.z());
    }

    while (!expand_queue.empty())
    {
        const openvdb::Coord c = expand_queue.front();
        expand_queue.pop();
        for (int i = 0; i < 26; ++i)
        {
            const openvdb::Coord nb = c.offsetBy(d26[i][0], d26[i][1], d26[i][2]);
            if (!frontier_acc.isValueOn(nb))
            {
                continue;
            }
            if (cluster_acc.isValueOn(nb))
            {
                continue;
            }

            cluster_acc.setValueOn(nb);
            expand_queue.push(nb);
            coords.push_back(nb);

            const auto pw = tf.indexToWorld(nb);
            points.emplace_back(pw.x(), pw.y(), pw.z());
        }
    }

    if (coords.size() < static_cast<size_t>(min_cluster_size_))
    {
        for (const auto &c : coords)
            cluster_acc.setValueOff(c);
        return;
    }

    FrontierCluster fc;
    fc.cell_coords_ = std::move(coords);
    fc.cells_ = std::move(points);
    compute_cluster_info(fc);
    tmp_frontiers_.push_back(std::move(fc));

    return;
}

void FrontierManager::compute_cluster_info(FrontierCluster &cluster)
{
    // Empty cluster
    if (cluster.cell_coords_.empty() || cluster.cells_.empty())
    {
        cluster.coord_box_.reset();
        cluster.centroid_.setZero();
        cluster.box_min_.setZero();
        cluster.box_max_.setZero();
        return;
    }

    // Bounding box
    openvdb::CoordBBox box(cluster.cell_coords_.front(), cluster.cell_coords_.front());
    for (size_t i = 1; i < cluster.cell_coords_.size(); ++i)
    {
        box.expand(cluster.cell_coords_[i]);
    }
    cluster.coord_box_ = box;

    // World-space bounding box
    const auto &tf = grid_clustered_frontier_->transform();
    const openvdb::Vec3d wmin = tf.indexToWorld(cluster.coord_box_.min());
    const openvdb::Vec3d wmax = tf.indexToWorld(cluster.coord_box_.max());

    cluster.box_min_ = Eigen::Vector3d(wmin.x(), wmin.y(), wmin.z());
    cluster.box_max_ = Eigen::Vector3d(wmax.x(), wmax.y(), wmax.z());

    // Centroid
    cluster.centroid_.setZero();
    for (const auto &p : cluster.cells_)
    {
        cluster.centroid_ += p;
    }
    cluster.centroid_ /= static_cast<double>(cluster.cells_.size());

    // Downsample
    downsample(cluster.cells_, cluster.filtered_cells_);
}

void FrontierManager::downsample(const std::vector<Eigen::Vector3d> &cluster_in,
                                 std::vector<Eigen::Vector3d> &cluster_out)
{
    // downsamping cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.reserve(cluster_in.size());

    for (auto &cell : cluster_in)
    {
        cloud->points.emplace_back(cell[0], cell[1], cell[2]);
    }

    const double voxel_size = grid_clustered_frontier_->voxelSize()[0]; // == VOX_SIZE
    const double leaf_size = voxel_size * down_sample_rate_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*cloudf);

    cluster_out.clear();
    cluster_out.reserve(cloudf->points.size());
    for (const auto &pt : cloudf->points)
    {
        cluster_out.emplace_back(pt.x, pt.y, pt.z);
    }
}

void FrontierManager::split_large_frontiers(std::list<FrontierCluster> &clusters_in)
{
    std::list<FrontierCluster> splits, tmps;

    for (auto it = clusters_in.begin(); it != clusters_in.end(); ++it)
    {
        // Can split
        if (split_horizontally(*it, splits))
        {
            tmps.insert(tmps.end(), splits.begin(), splits.end());
            splits.clear(); // 和原作者位置一致：用完再清
        }
        else
        {
            tmps.push_back(*it);
        }
    }
    clusters_in.swap(tmps);
}

bool FrontierManager::split_horizontally(const FrontierCluster &frontier,
                                         std::list<FrontierCluster> &out_splits)
{
    if (frontier.filtered_cells_.empty())
    {
        return false;
    }

    // Need to be splitted?
    const Eigen::Vector2d mean = frontier.centroid_.head<2>();
    bool need_split = false;
    for (const auto &cell : frontier.filtered_cells_)
    {
        if ((cell.head<2>() - mean).norm() > cluster_size_xy_)
        {
            need_split = true;
            break;
        }
    }
    if (!need_split)
    {
        return false;
    }

    // 2) 2x2 cov
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto &cell : frontier.filtered_cells_)
    {
        const Eigen::Vector2d d = cell.head<2>() - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(frontier.filtered_cells_.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    int max_idx;
    es.eigenvalues().maxCoeff(&max_idx);
    const Eigen::Vector2d first_pc = es.eigenvectors().col(max_idx);

    // 3) Split, assign points
    FrontierCluster ftr1, ftr2;
    const size_t N = frontier.cells_.size();
    ftr1.cells_.reserve(N / 2);
    ftr1.cell_coords_.reserve(N / 2);
    ftr2.cells_.reserve(N / 2);
    ftr2.cell_coords_.reserve(N / 2);

    for (size_t i = 0; i < N; ++i)
    {
        const Eigen::Vector3d &p = frontier.cells_[i];
        const double s = (p.head<2>() - mean).dot(first_pc);
        if (s >= 0)
        {
            ftr1.cells_.push_back(p);
            ftr1.cell_coords_.push_back(frontier.cell_coords_[i]);
        }
        else
        {
            ftr2.cells_.push_back(p);
            ftr2.cell_coords_.push_back(frontier.cell_coords_[i]);
        }
    }
    if (ftr1.cells_.empty() || ftr2.cells_.empty())
    {
        return false;
    }

    compute_cluster_info(ftr1);
    compute_cluster_info(ftr2);

    // Recursive
    std::list<FrontierCluster> splits2;
    if (split_horizontally(ftr1, splits2))
    {
        out_splits.insert(out_splits.end(), splits2.begin(), splits2.end());
        splits2.clear();
    }
    else
    {
        out_splits.push_back(std::move(ftr1));
    }

    if (split_horizontally(ftr2, splits2))
    {
        out_splits.insert(out_splits.end(), splits2.begin(), splits2.end());
        splits2.clear();
    }
    else
    {
        out_splits.push_back(std::move(ftr2));
    }

    return true;
}

bool FrontierManager::is_viewpoint_safe_edt(const Eigen::Vector3d &pos_world,
                                            const DynamicVDBEDT &edt)
{
    const auto &tf = grid_clustered_frontier_->transform();
    openvdb::Vec3d idx = tf.worldToIndex(openvdb::Vec3d(pos_world.x(),
                                                        pos_world.y(),
                                                        pos_world.z()));
    openvdb::Coord ijk = openvdb::Coord::round(idx);

    // EDT returns voxel^2 distance
    double sq = edt.query_sq_distance(ijk);
    if (sq < 0.0)
    {
        // no value, not safe
        return false;
    }

    return (sq >= safe_sq_dist_);
}

void FrontierManager::compute_frontiers_to_visit(const openvdb::FloatGrid::ConstAccessor &occ_acc,
                                                 const std::shared_ptr<DynamicVDBEDT> &dist_map)
{
    // Mark the postion of the first new cluster
    first_new_cluster_ = frontiers_.end();
    int new_num = 0, new_dormant = 0;

    for (auto &tmp_ftr : tmp_frontiers_)
    {
        sample_viewpoints(tmp_ftr, occ_acc, dist_map);
        if (!tmp_ftr.viewpoints_.empty())
        {
            ++new_num;
            auto inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);

            // Put the best viewpoint in the front
            std::sort(inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
                      [](const Viewpoint &a, const Viewpoint &b)
                      { return a.visib_num_ > b.visib_num_; });

            // Mark the first inserted cluster
            if (first_new_cluster_ == frontiers_.end())
            {
                first_new_cluster_ = inserted;
            }
        }
        else
        {
            dormant_frontiers_.push_back(tmp_ftr);
            ++new_dormant;
        }
    }

    int idx = 0;
    for (auto &ft : frontiers_)
    {
        ft.id_ = idx++;
    }

    tmp_frontiers_.clear();
}

void FrontierManager::sample_viewpoints(FrontierCluster &frontier,
                                        const openvdb::FloatGrid::ConstAccessor &occ_acc,
                                        const std::shared_ptr<DynamicVDBEDT> &dist_map)
{
    frontier.viewpoints_.clear();

    if (frontier.filtered_cells_.empty())
    {
        return;
    }

    const Eigen::Vector3d &avg = frontier.centroid_;

    if (candidate_rnum_ <= 0)
    {
        return;
    }

    const double rmin = viewpoint_rmin_;
    const double rmax = viewpoint_rmax_;
    if (rmax < rmin)
    {
        return;
    }

    const double dr = (rmax - rmin) / static_cast<double>(candidate_rnum_);

    for (double rc = rmin; rc <= rmax + 1e-3; rc += dr)
    {
        for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_)
        {
            const Eigen::Vector3d sample_pos = avg + rc * Eigen::Vector3d(std::cos(phi), std::sin(phi), 0.0);

            // viewpoint itself is safe
            if (!is_viewpoint_safe_edt(sample_pos, *dist_map))
            {
                continue;
            }

            // get avg
            const auto &cells = frontier.filtered_cells_;
            if (cells.empty())
            {
                continue;
            }

            Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();

            double avg_yaw = 0.0;
            for (size_t i = 1; i < cells.size(); ++i)
            {
                Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
                double dotv = dir.dot(ref_dir);
                if (dotv > 1.0)
                {
                    dotv = 1.0;
                }
                if (dotv < -1.0)
                {
                    dotv = -1.0;
                }

                double yaw = std::acos(dotv);
                if (ref_dir.cross(dir).z() < 0.0)
                {
                    yaw = -yaw;
                }
                avg_yaw += yaw;
            }

            if (!cells.empty())
            {
                avg_yaw = avg_yaw / static_cast<double>(cells.size());
            }
            avg_yaw += std::atan2(ref_dir.y(), ref_dir.x());

            // wrap [-pi, pi]
            while (avg_yaw < -M_PI)
            {
                avg_yaw += 2.0 * M_PI;
            }
            while (avg_yaw > M_PI)
            {
                avg_yaw -= 2.0 * M_PI;
            }

            // 可见性计数
            const int visib_num = count_visible_cells(sample_pos, cells, occ_acc);

            if (visib_num > min_visib_num_)
            {
                Viewpoint vp;
                vp.pos_ = sample_pos;
                vp.yaw_ = avg_yaw;
                vp.visib_num_ = visib_num;
                frontier.viewpoints_.push_back(vp);
            }
            else
            {
                // 视点覆盖不足，忽略
            }
        }
    }
}

int FrontierManager::count_visible_cells(
    const Eigen::Vector3d& pos,
    const std::vector<Eigen::Vector3d>& cluster_cells,
    const openvdb::FloatGrid::ConstAccessor& occ_acc)
{
    double L_THRESH = 0;
    int visib_num = 0;

    // 使用 clustered frontier grid 的 transform（与你的 occ grid 对齐）
    const openvdb::math::Transform& tf = grid_clustered_frontier_->transform();

    const openvdb::Vec3d origin_w(pos.x(), pos.y(), pos.z());
    const openvdb::Vec3d origin_ijk = tf.worldToIndex(origin_w);

    const float occ_thresh = static_cast<float>(L_THRESH);  // 或者你自己的阈值成员

    for (const auto& cell_w : cluster_cells)
    {
        // xyz - ijk
        const openvdb::Vec3d target_w(cell_w.x(), cell_w.y(), cell_w.z());
        openvdb::Vec3d target_ijk = tf.worldToIndex(target_w);

        openvdb::Vec3d dir = target_ijk - origin_ijk;
        const double len = dir.length();

        if (len <= 1e-6)
        {
            ++visib_num;
            continue;
        }

        dir.normalize();

        // ray + DDA
        openvdb::math::Ray<double> ray(origin_ijk, dir);
        openvdb::math::DDA<openvdb::math::Ray<double>, 0> dda(
            ray, 0.0, len);

        bool visible = true;

        // ray tracing
        for (;;)
        {
            const openvdb::Coord ijk = dda.voxel();

            float val = 0.0f;
            const bool known = occ_acc.probeValue(ijk, val);

            // unknown
            if (!known)
            {
                visible = false;
                break;
            }

            // occupied
            if (val >= occ_thresh)
            {
                visible = false;
                break;
            }

            dda.step();

            if (!(dda.time() < dda.maxTime()))
            {
                break;
            }
        }

        if (visible)
        {
            ++visib_num;
        }
    }

    return visib_num;
}

void FrontierManager::updateFrontierCostMatrix()
{
    std::cout << "cost mat size before remove: " << std::endl;
    for (auto &ftr : frontiers_)
    {
        std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
    }
    std::cout << std::endl;

    std::cout << "cost mat size remove: " << std::endl;
    if (!removed_ids_.empty())
    {
        // 对所有“旧 frontier”（first_new_cluster_ 之前的）删掉被移除 cluster 对应的列
        for (auto it = frontiers_.begin(); it != first_new_cluster_; ++it)
        {
            auto cost_iter = it->costs_.begin();
            auto path_iter = it->paths_.begin();
            int iter_idx = 0;

            for (int rid : removed_ids_)
            {
                // move the iterator to the delete position
                while (iter_idx < rid && cost_iter != it->costs_.end())
                {
                    ++cost_iter;
                    ++path_iter;
                    ++iter_idx;
                }
                if (cost_iter == it->costs_.end())
                    break;

                cost_iter = it->costs_.erase(cost_iter);
                path_iter = it->paths_.erase(path_iter);
                // 注意：这里不 ++iter_idx，因为 after erase, current position is still rid
            }
            std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
        }
        removed_ids_.clear();
    }
    std::cout << std::endl;

    // 小工具：给两个 cluster 更新双向 cost / path
    auto updateCost = [this](const std::list<FrontierCluster>::iterator &it1,
                             const std::list<FrontierCluster>::iterator &it2)
    {
        std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";

        // Viewpoints first, if no viewpoints, use centroid
        const Eigen::Vector3d p1 = it1->viewpoints_.empty()
                                       ? it1->centroid_
                                       : it1->viewpoints_.front().pos_;
        const Eigen::Vector3d p2 = it2->viewpoints_.empty()
                                       ? it2->centroid_
                                       : it2->viewpoints_.front().pos_;

        std::vector<Eigen::Vector3d> path_12;
        double cost_12 = computeCost(p1, p2, path_12); // distance only

        // it1 -> it2
        it1->costs_.push_back(cost_12);
        it1->paths_.push_back(path_12);

        // it2 -> it1
        std::reverse(path_12.begin(), path_12.end());
        it2->costs_.push_back(cost_12);
        it2->paths_.push_back(path_12);
    };

    // ========== old - new ==========
    for (auto it_old = frontiers_.begin(); it_old != first_new_cluster_; ++it_old)
    {
        for (auto it_new = first_new_cluster_; it_new != frontiers_.end(); ++it_new)
        {
            updateCost(it_old, it_new);
        }
    }

    // ========== new - new ==========
    if (first_new_cluster_ != frontiers_.end())
    {
        for (auto it_i = first_new_cluster_; it_i != frontiers_.end(); ++it_i)
        {
            auto it_j = it_i;
            ++it_j;
            for (; it_j != frontiers_.end(); ++it_j)
            {
                updateCost(it_i, it_j);
            }
        }
    }

    std::cout << std::endl;
}

double FrontierManager::computeCost(const Eigen::Vector3d &p1,
                                    const Eigen::Vector3d &p2,
                                    std::vector<Eigen::Vector3d> &path)
{
    path.clear();
    path.push_back(p1);
    path.push_back(p2);

    return (p2 - p1).norm();
}