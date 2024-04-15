#pragma once
#include "ieskf.h"
#include "commons.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

namespace lio
{
    enum LIOStatus
    {
        IMU_INIT,
        MAP_INIT,
        LIO_MAPPING

    };
    struct LIOConfig
    {
        int opti_max_iter = 5;
        double na = 0.01;
        double ng = 0.01;
        double nba = 0.0001;
        double nbg = 0.0001;
        double scan_resolution = 0.1;
        double map_resolution = 0.5;
        int imu_init_num = 20;
        Eigen::Matrix3d r_il = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_il = Eigen::Vector3d::Zero();
        bool gravity_align = true;
    };
    struct LIODataGroup
    {
        IMUData last_imu;
        std::vector<IMUData> imu_cache;
        std::vector<Pose> imu_poses_cache;
        Eigen::Vector3d last_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d last_gyro = Eigen::Vector3d::Zero();
        double last_cloud_end_time = 0.0;
        double gravity_norm;
        kf::Matrix12d Q = kf::Matrix12d::Identity();
    };

    class LIOBuilder
    {
    public:
        LIOBuilder() = default;

        void loadConfig(LIOConfig &_config);

        bool initializeImu(std::vector<IMUData> &imus);

        void undistortCloud(SyncPackage &package);

        void process(SyncPackage &package);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToWorld(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToBody(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

    public:
        kf::IESKF kf;
        LIOConfig config;
        LIODataGroup data_group;
        LIOStatus status = LIOStatus::IMU_INIT;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar_cloud;
        pcl::VoxelGrid<pcl::PointXYZINormal> scan_filter;
    };

} // namespace lio
