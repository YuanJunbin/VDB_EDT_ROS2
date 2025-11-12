#ifndef _FRONTIER_CLUSTER_H_
#define _FRONTIER_CLUSTER_H_

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>

struct FrontierCluster
{
    // Frontier voxels in the cluster
    std::vector<Eigen::Vector3d> cells_;
    std::vector<openvdb::Coord> cell_coords_;

    // Down-sampled voxels for viewpoint extraction
    std::vector<Eigen::Vector3d> filtered_cells_;

    // Cluster centroid
    Eigen::Vector3d centroid_;

    // Idx of cluster
    int id_;

    // Viewpoints that can cover the cluster
    // std::vector<Viewpoint> viewpoints_;

    // Bounding box of cluster, center & 1/2 side length
    Eigen::Vector3d box_min_, box_max_;
    openvdb::CoordBBox coord_box_;

    // Path and cost from this cluster to other clusters
    std::list<std::vector<Eigen::Vector3d>> paths_;
    std::list<double> costs_;
};

class FrontierManager
{
private:
    std::list<FrontierCluster> frontiers_, dormant_frontiers_, tmp_frontiers_;
    openvdb::BoolGrid::Ptr grid_clustered_frontier_;

    // Remove old
    bool cluster_overlap_box(const FrontierCluster &cluster,
                             const openvdb::CoordBBox &expanded_box);

    bool any_old_frontier_off(const FrontierCluster &cluster,
                              const openvdb::BoolGrid::ConstAccessor &grid_acc);

    void unflag_changed_cluster(const FrontierCluster &cluster);

    void reset_changed_clusters(const openvdb::CoordBBox &expanded_box,
                                const openvdb::BoolGrid::ConstAccessor &frontier_acc);
    std::vector<int> removed_ids_;

    // Add new
    void frontier_clustering(const openvdb::BoolGrid::Ptr &grid_frontier,
                             const openvdb::CoordBBox &expanded_box);
    void expand_frontier(const openvdb::Coord &seed,
                         openvdb::BoolGrid::ConstAccessor &frontier_acc,
                         openvdb::BoolGrid::Accessor &cluster_acc);
    void compute_cluster_info(FrontierCluster &cluster);

    // min number of points in a cluster
    int min_cluster_size_;

    void downsample(const std::vector<Eigen::Vector3d> &cluster_in,
                    std::vector<Eigen::Vector3d> &cluster_out);
    double down_sample_rate_;

    void split_large_frontiers(std::list<FrontierCluster> &clusters_in);
    bool split_horizontally(const FrontierCluster &frontier,
                            std::list<FrontierCluster> &out_splits);
    // metric size threshold of a cluster, help determine split/merge
    double cluster_size_xy_;
    

public:
    // Search box, expanded from update box
    openvdb::CoordBBox expand_update_box(const openvdb::CoordBBox &update_box,
                                         double voxel_size);
    double expand_x_, expand_y_, expand_z_;
};

#endif