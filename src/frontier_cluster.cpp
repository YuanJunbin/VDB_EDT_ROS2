#include "vdb_edt/frontier_cluster.h"

#include <queue>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>

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
