#pragma once
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lio
{
    Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);

    float sq_dist(const pcl::PointXYZINormal &p1, const pcl::PointXYZINormal &p2);

    struct IMUData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        double timestamp;
        IMUData() = default;
        IMUData(const Eigen::Vector3d &a, const Eigen::Vector3d &g, double &d) : acc(a), gyro(g), timestamp(d) {}
    };

    struct SyncPackage
    {
        std::vector<lio::IMUData> imus;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
        double cloud_start_time = 0.0;
        double cloud_end_time = 0.0;
    };

    struct Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        Eigen::Matrix3d rot;
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
        Pose();
        Pose(double t, Eigen::Vector3d a, Eigen::Vector3d g, Eigen::Vector3d v, Eigen::Vector3d p, Eigen::Matrix3d r)
            : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
        double offset;
    };

    void calcBodyCov(Eigen::Vector3d &pb, const double &range_inc, const double &degree_inc, Eigen::Matrix3d &cov);
}
