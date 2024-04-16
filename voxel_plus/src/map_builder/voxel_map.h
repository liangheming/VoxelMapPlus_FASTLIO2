#pragma once
#include <list>
#include <vector>
#include <memory>
#include <cstdint>
#include <Eigen/Eigen>
#include <unordered_map>
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
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d center;
        Eigen::Vector3d normal;
        Eigen::Vector3d x_normal;
        Eigen::Vector3d y_normal;
        Eigen::Matrix3d covariance;
        Eigen::Vector3d eigens;
        Eigen::Matrix<double, 6, 6> plane_cov;
        bool is_valid;
        int points_size;
    };

    struct ResidualData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d plane_center;
        Eigen::Vector3d plane_norm;
        Eigen::Matrix<double, 6, 6> plane_cov;
        Eigen::Matrix3d pcov;
        Eigen::Matrix3d cov;
        Eigen::Vector3d point_lidar;
        Eigen::Vector3d point_world;
        bool is_valid = false;
        bool from_near = false;
        int current_layer = 0;
        double sigma_num = 3.0;
        double residual = 0.0;
    };
    class OctoTree
    {
    public:
        OctoTree(int _max_layer, int _layer, std::vector<int> _update_size_threshes, int _max_point_thresh, double _plane_thresh);

        void insert(const std::vector<PointWithCov> &input_points);

        void initialize_tree();

        void build_plane(const std::vector<PointWithCov> &points);

        void split_tree();
        int subIndex(const PointWithCov &pv, int *xyz);

    public:
        double quater_length;
        Eigen::Vector3d center;
        Plane plane;
        int layer;
        int max_layer;
        bool is_leave;
        bool is_initialized;
        bool update_enable;
        std::vector<std::shared_ptr<OctoTree>> leaves;
        std::vector<PointWithCov> temp_points;
        int update_size_thresh;
        std::vector<int> update_size_threshes;
        double plane_thresh;
        int max_point_thresh;
        int update_size_thresh_for_new;
        int all_point_num;
        int new_point_num;
    };
    struct VoxelValue
    {
        std::list<VoxelKey>::iterator it;
        std::shared_ptr<OctoTree> tree;
    };

    enum SubVoxelType
    {
        INSERT,
        UPDATE
    };

    struct VoxelGrid
    {
        SubVoxelType type;
        std::list<VoxelKey>::iterator it;
        std::vector<PointWithCov> points;
    };
    typedef std::unordered_map<VoxelKey, VoxelValue, VoxelKey::Hasher> FeatMap;
    typedef std::unordered_map<VoxelKey, VoxelGrid, VoxelKey::Hasher> SubMap;
    class VoxelMap
    {
    public:
        VoxelMap(double _voxel_size, int _max_layer, std::vector<int> &_update_size_threshes, int _max_point_thresh, double _plane_thresh, int _capacity = 5000000)
            : voxel_size(_voxel_size), max_layer(_max_layer), update_size_threshes(_update_size_threshes), max_point_thresh(_max_point_thresh), plane_thresh(_plane_thresh), capacity(_capacity)
        {
            feat_map.clear();
            sub_map.clear();
            cache.clear();
        }

        void insert(const std::vector<PointWithCov> &input_points);

        void pack(const std::vector<PointWithCov> &input_points);

        VoxelKey index(const Eigen::Vector3d &point);

        void buildResidual(ResidualData &info, std::shared_ptr<OctoTree> oct_tree);

    public:
        FeatMap feat_map;
        SubMap sub_map;
        std::list<VoxelKey> cache;
        double voxel_size;
        int max_layer;
        std::vector<int> update_size_threshes;
        int max_point_thresh;
        double plane_thresh;
        int capacity;
    };
} // namespace lio
