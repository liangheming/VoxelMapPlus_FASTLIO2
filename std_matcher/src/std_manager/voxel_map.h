#pragma once
#include "utils.h"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <queue>
#include <unordered_set>

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
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double plane_thresh;
        int min_num_thresh;
        Eigen::Vector3d sum;
        Eigen::Matrix3d ppt;
        Eigen::Matrix3d norms;
        Eigen::Vector3d lamdas;
        double is_plane;
        pcl::PointCloud<pcl::PointXYZINormal> clouds;
    };

    typedef std::unordered_map<VoxelKey, std::shared_ptr<VoxelGrid>, VoxelKey::Hasher> FeatMap;

    struct Plane
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d ppt = Eigen::Matrix3d::Zero();
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d lamdas = Eigen::Vector3d::Zero();
        Eigen::Matrix3d norms = Eigen::Matrix3d::Zero();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr sur_cloud;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_cloud;
        std::unordered_set<VoxelKey, VoxelKey::Hasher> corner_voxels;
        int num = 0;
    };

    class VoxelMap
    {
    public:
        VoxelMap(double _voxel_size, double _plane_thesh, int _min_num_thresh);

        VoxelKey index(const pcl::PointXYZINormal &p);

        static VoxelKey index(double x, double y, double z, double resolution);

        static std::vector<VoxelKey> nears(const VoxelKey &center, int range);

        static std::vector<VoxelKey> sixNears(const VoxelKey &center);

        void build(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        void mergePlanes();

        void updateMergeThresh(double _merge_thresh) { merge_thresh = _merge_thresh; }

        void reset();

        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> coloredPlaneCloud(bool with_corner = false);

    public:
        double plane_thresh;
        double voxel_size;
        int min_num_thresh;
        FeatMap voxels;
        std::vector<VoxelKey> plane_voxels;
        std::vector<Plane> planes;
        double merge_thresh = 0.1;
    };

    struct STDDescriptor
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t frame_id;
        Eigen::Vector3d side_length;
        Eigen::Vector3d angle;
        Eigen::Vector3d center;
        Eigen::Vector3d vertex_a;
        Eigen::Vector3d vertex_b;
        Eigen::Vector3d vertex_c;
        Eigen::Vector3d attached;
        Eigen::Matrix3d norms;
    };

    struct Grid2d
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector2d xy = Eigen::Vector2d::Zero();
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d norms = Eigen::Matrix3d::Zero();
        int num = 0;
        bool flag = true;
    };
    
    class STDExtractor
    {
    public:
        STDExtractor(std::shared_ptr<VoxelMap> _voxel_map);

        void extract(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, uint64_t frame_id, std::vector<STDDescriptor> &descs);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr projectCornerNMS(const Plane &plane);

        static std::vector<VoxelKey> nears2d(const VoxelKey &center, int range);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr nms3D(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud);

        void buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc);

    public:
        std::shared_ptr<VoxelMap> voxel_map;
        double image_resolution = 0.25;
        int nms_range = 5;
        double nms_3d_range = 1.0;
        int max_corner_num = 100;
        int descriptor_near_num = 15;
        double std_side_resolution = 0.2;
        double min_dis_threshold = 1.0;
        double max_dis_threshold = 30.0;
        double project_min_dis = 0.0;
        double project_max_dis = 5.0;
    };
} // namespace stdes
