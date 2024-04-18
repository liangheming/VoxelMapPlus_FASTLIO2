#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <iostream>

#define HASH_P 116101
#define MAX_N 10000000000

namespace lio
{
    class VoxelKey
    {
    public:
        int64_t x, y, z;

        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }

        struct Hasher
        {
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    struct PointWithCov
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point;
        Eigen::Matrix3d cov;
    };

    struct Plane
    {
        bool is_plane = false;
        bool is_init = false;
        bool is_root_plane = true;

        double xx = 0.0;
        double yy = 0.0;
        double zz = 0.0;
        double xy = 0.0;
        double xz = 0.0;
        double yz = 0.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        int n = 0;

        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d ppt = Eigen::Matrix3d::Zero();
        Eigen::Matrix4d plane_cov = Eigen::Matrix4d::Zero();
        Eigen::Vector4d plane_param = Eigen::Vector4d::Zero();
    };

    class VoxelMap;

    class UnionFindNode
    {
    public:
        UnionFindNode(double _plane_thresh, int _update_size_thresh, int _max_point_thresh, Eigen::Vector3d &_voxel_center, VoxelMap *_map);
        void updatePlane();
        void addToPlane(const PointWithCov &pv);
        void push(const PointWithCov &pv);
        void emplace(const PointWithCov &pv);
        bool isPlane() { return plane->is_plane; }
        bool isInitialized() { return plane->is_init; }

    public:
        UnionFindNode *root_node;
        VoxelMap *map;
        std::shared_ptr<Plane> plane;
        std::vector<PointWithCov> temp_points;
        Eigen::Vector3d voxel_center;
        // int total_point_num;
        int newly_added_num;
        int update_size_thresh;
        int max_point_thresh;
        bool update_enable;
        double plane_thesh;
    };

    typedef std::unordered_map<VoxelKey, UnionFindNode *, VoxelKey::Hasher> FeatMap;

    class VoxelMap
    {
    public:
        VoxelMap(double _voxel_size, double _plane_thresh, int _update_size_thresh, int _max_point_thresh);

        VoxelKey index(const Eigen::Vector3d &point);

        void build(std::vector<PointWithCov> &pvs);

        void update(std::vector<PointWithCov> &pvs);

        ~VoxelMap();

    public:
        FeatMap feat_map;
        double voxel_size;
        double plane_thresh;
        int update_size_thresh;
        int max_point_thresh;
    };

} // namespace lio
