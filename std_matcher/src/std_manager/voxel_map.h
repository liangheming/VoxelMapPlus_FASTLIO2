#pragma once
#include "utils.h"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <queue>

#define HASH_P 116101
#define MAX_N 10000000000

namespace stdes
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

    class VoxelGrid
    {
    public:
        VoxelGrid(double _plane_thesh, int _min_num_thresh);

        void buildPlane();

        void addPoint(const pcl::PointXYZINormal &p);

    public:
        double plane_thresh;
        int min_num_thresh;
        Eigen::Vector3d mean;
        Eigen::Matrix3d ppt;
        Eigen::Matrix3d norms;
        Eigen::Vector3d lamdas;
        double is_plane;
        pcl::PointCloud<pcl::PointXYZINormal> clouds;
    };

    typedef std::unordered_map<VoxelKey, std::shared_ptr<VoxelGrid>, VoxelKey::Hasher> FeatMap;

    struct Plane
    {
        Eigen::Vector3d lamdas;
        Eigen::Matrix3d norms;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr sur_cloud;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_cloud;
        int num = 0;
    };

    class VoxelMap
    {
    public:
        VoxelMap(double _voxel_size, double _plane_thesh, int _min_num_thresh);

        VoxelKey index(const pcl::PointXYZINormal &p);

        static std::vector<VoxelKey> nears(const VoxelKey &center, int range);

        static std::vector<VoxelKey> sixNears(const VoxelKey &center);

        void build(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        void mergePlanes();

        void updateMergeThresh(double _merge_thresh) { merge_thresh = _merge_thresh; }

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr coloredPlaneCloud();

    public:
        double plane_thresh;
        double voxel_size;

        int min_num_thresh;
        FeatMap voxels;
        std::vector<VoxelKey> plane_voxels;
        std::vector<Plane> planes;
        double merge_thresh = 0.1;
    };
} // namespace stdes
