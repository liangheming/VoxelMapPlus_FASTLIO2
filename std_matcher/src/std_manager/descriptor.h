#pragma once
#include <vector>
#include <cstdint>
#include <Eigen/Eigen>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sophus/so3.hpp>

#define HASH_P 116101
#define MAX_N 10000000000
namespace std_desc
{
    struct MPoint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector4d xyzi;
        int count;
    };
    void voxelFilter(pcl::PointCloud<pcl::PointXYZINormal>::Ptr in, double voxel_size);

    Eigen::Matrix3d skew(const Eigen::Vector3d &vec);

    class VoxelKey
    {
    public:
        int64_t x, y, z;

        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }

        static VoxelKey index(double x, double y, double z, double resolution, double bias = 0.0);

        static std::vector<VoxelKey> rangeNear(VoxelKey center, int range, bool exclude_self);

        struct Hasher
        {
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    struct Config
    {
        int voxel_min_point = 10;
        double voxel_size = 1.0;
        double voxel_plane_thresh = 0.01;
        double norm_merge_thresh = 0.1;

        double proj_2d_resolution = 0.25;
        double proj_min_dis = 0.0001;
        double proj_max_dis = 5.0;

        int nms_2d_range = 5;
        double nms_3d_range = 2.0;
        double corner_thresh = 10.0;
        int max_corner_num = 100;
        double min_side_len = 2.0;
        double max_side_len = 30.0;
        int desc_search_range = 15;

        double side_resolution = 0.2;
        double rough_dis_threshold = 0.03;
        int skip_near_num = 50;
        double vertex_diff_threshold = 0.7;
        int candidate_num = 50;
        double verify_dis_thresh = 3.0;
        double geo_verify_dis_thresh = 0.3;
        double icp_thresh = 0.5;

        double iter_eps = 1e-3;
        size_t max_iter = 10;
    };

    struct VoxelNode
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d sum;
        Eigen::Vector3d mean;
        Eigen::Matrix3d ppt;
        Eigen::Vector3d norm;
        Eigen::Matrix3d norms;
        Eigen::Vector3d lamdas;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
        bool is_plane = false;
        bool is_valid = false;
        bool is_projected = false;
        std::vector<Eigen::Vector3d> projected_norms;
        bool connect[6] = {false, false, false, false, false, false};
        bool connect_check[6] = {false, false, false, false, false, false};
        VoxelNode *connect_nodes[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    };

    struct STDDescriptor
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d side_length;
        Eigen::Vector3d angle;
        Eigen::Vector3d center;
        Eigen::Vector3d vertex_a;
        Eigen::Vector3d vertex_b;
        Eigen::Vector3d vertex_c;
        Eigen::Vector3d attached;
        Eigen::Matrix3d norms;
        uint64_t id;
    };
    struct STDFeature
    {
        double stamp;
        uint64_t id;
        bool valid;
        std::vector<STDDescriptor> descs;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
    };
    struct STDMatch
    {
        uint64_t match_id;
        std::vector<std::pair<STDDescriptor, STDDescriptor>> match_pairs;
    };
    struct LoopResult
    {
        bool valid = false;
        uint64_t match_id;
        double match_score;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        std::vector<std::pair<STDDescriptor, STDDescriptor>> match_pairs;
    };

    class STDManager
    {
    public:
        STDManager(Config &_config) : config(_config) {}

        STDFeature extract(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        void buildVoxels(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr extractCorners();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr nms2d(const Eigen::Vector3d &mean, const Eigen::Vector3d &norm, std::vector<Eigen::Vector3d> &proj_points);

        void nms3d(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud);

        void buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc);

        void buildConnections();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr currentPlaneCloud();

        void insert(STDFeature &feature);

        void triangleSolver(std::pair<STDDescriptor, STDDescriptor> &pair, Eigen::Matrix3d &rot, Eigen::Vector3d &trans);

        void selectCanidates(STDFeature &feature, std::vector<STDMatch> &match_list);

        double verifyCandidate(STDMatch &matches,
                               Eigen::Matrix3d &rot,
                               Eigen::Vector3d &trans,
                               std::vector<std::pair<STDDescriptor, STDDescriptor>> &success_match_vec,
                               pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud);

        double verifyGeoPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                              pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                              const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans);

        double verifyGeoPlaneICP(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                 Eigen::Matrix3d &rot, Eigen::Vector3d &trans);

        LoopResult searchLoop(STDFeature &feature);

    public:
        Config config;
        static uint64_t frame_count;
        std::unordered_map<VoxelKey, VoxelNode, VoxelKey::Hasher> temp_voxels;
        std::unordered_map<VoxelKey, std::vector<STDDescriptor>, VoxelKey::Hasher> data_base;
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> cloud_vec;
    };
}