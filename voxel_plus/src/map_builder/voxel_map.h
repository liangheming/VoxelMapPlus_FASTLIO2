#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Eigen>
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
        Eigen::Matrix3d plane_cov;
        Eigen::Vector3d n_vec;

        double xx = 0.0;
        double yy = 0.0;
        double zz = 0.0;
        double xy = 0.0;
        double xz = 0.0;
        double yz = 0.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    };

    class UnionFindNode
    {
        public:
            UnionFindNode();
        public:
            
    };

} // namespace lio
