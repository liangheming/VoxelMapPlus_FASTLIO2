#include "lio_builder.h"

namespace lio
{
    void LIOBuilder::loadConfig(LIOConfig &_config)
    {
        config = _config;
        status = LIOStatus::IMU_INIT;
        data_group.Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * config.ng;
        data_group.Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * config.na;
        data_group.Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * config.nbg;
        data_group.Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * config.nba;
        if (config.scan_resolution > 0.0)
            scan_filter.setLeafSize(config.scan_resolution, config.scan_resolution, config.scan_resolution);
        lidar_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        // kf.set_share_function(
        //     [this](kf::State &s, kf::SharedState &d)
        //     { sharedUpdateFunc(s, d); });
    }
    bool LIOBuilder::initializeImu(std::vector<IMUData> &imus)
    {
        data_group.imu_cache.insert(data_group.imu_cache.end(), imus.begin(), imus.end());
        if (data_group.imu_cache.size() < config.imu_init_num)
            return false;
        Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
        for (const auto &imu : data_group.imu_cache)
        {
            acc_mean += imu.acc;
            gyro_mean += imu.gyro;
        }
        acc_mean /= static_cast<double>(data_group.imu_cache.size());
        gyro_mean /= static_cast<double>(data_group.imu_cache.size());
        data_group.gravity_norm = acc_mean.norm();
        kf.x().rot_ext = config.r_il;
        kf.x().pos_ext = config.p_il;
        kf.x().bg = gyro_mean;
        if (config.gravity_align)
        {
            kf.x().rot = (Eigen::Quaterniond::FromTwoVectors((-acc_mean).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).matrix());
            kf.x().initG(Eigen::Vector3d(0, 0, -1.0));
        }
        else
        {
            kf.x().initG(-acc_mean);
        }
        kf.P().setIdentity();
        kf.P().block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 0.00001;
        kf.P().block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 0.00001;
        kf.P().block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 0.0001;
        kf.P().block<3, 3>(18, 18) = Eigen::Matrix3d::Identity() * 0.0001;
        kf.P().block<2, 2>(21, 21) = Eigen::Matrix2d::Identity() * 0.00001;
        data_group.last_imu = imus.back();
        return true;
    }

    void LIOBuilder::undistortCloud(SyncPackage &package)
    {
        data_group.imu_cache.clear();
        data_group.imu_cache.push_back(data_group.last_imu);
        data_group.imu_cache.insert(data_group.imu_cache.end(), package.imus.begin(), package.imus.end());

        const double imu_time_begin = data_group.imu_cache.front().timestamp;
        const double imu_time_end = data_group.imu_cache.back().timestamp;
        const double cloud_time_begin = package.cloud_start_time;
        const double cloud_time_end = package.cloud_end_time;
        std::sort(package.cloud->points.begin(), package.cloud->points.end(), [](pcl::PointXYZINormal &p1, pcl::PointXYZINormal &p2) -> bool
                  { return p1.curvature < p2.curvature; });

        data_group.imu_poses_cache.clear();
        data_group.imu_poses_cache.emplace_back(0.0, data_group.last_acc, data_group.last_gyro,
                                                kf.x().vel, kf.x().pos, kf.x().rot);

        Eigen::Vector3d acc_val, gyro_val;
        double dt = 0.0;
        kf::Input inp;

        for (auto it_imu = data_group.imu_cache.begin(); it_imu < (data_group.imu_cache.end() - 1); it_imu++)
        {
            IMUData &head = *it_imu;
            IMUData &tail = *(it_imu + 1);

            if (tail.timestamp < data_group.last_cloud_end_time)
                continue;
            gyro_val = 0.5 * (head.gyro + tail.gyro);
            acc_val = 0.5 * (head.acc + tail.acc);

            acc_val = acc_val * 9.81 / data_group.gravity_norm;

            if (head.timestamp < data_group.last_cloud_end_time)
                dt = tail.timestamp - data_group.last_cloud_end_time;
            else
                dt = tail.timestamp - head.timestamp;

            inp.acc = acc_val;
            inp.gyro = gyro_val;

            kf.predict(inp, dt, data_group.Q);

            data_group.last_gyro = gyro_val - kf.x().bg;
            data_group.last_acc = kf.x().rot * (acc_val - kf.x().ba) + kf.x().g;

            double offset = tail.timestamp - cloud_time_begin;
            data_group.imu_poses_cache.emplace_back(offset, data_group.last_acc, data_group.last_gyro, kf.x().vel, kf.x().pos, kf.x().rot);
        }

        dt = cloud_time_end - imu_time_end;
        kf.predict(inp, dt, data_group.Q);

        data_group.last_imu = package.imus.back();
        data_group.last_cloud_end_time = cloud_time_end;

        Eigen::Matrix3d cur_rot = kf.x().rot;
        Eigen::Vector3d cur_pos = kf.x().pos;
        Eigen::Matrix3d cur_rot_ext = kf.x().rot_ext;
        Eigen::Vector3d cur_pos_ext = kf.x().pos_ext;

        auto it_pcl = package.cloud->points.end() - 1;
        for (auto it_kp = data_group.imu_poses_cache.end() - 1; it_kp != data_group.imu_poses_cache.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;

            Eigen::Matrix3d imu_rot = head->rot;
            Eigen::Vector3d imu_pos = head->pos;
            Eigen::Vector3d imu_vel = head->vel;
            Eigen::Vector3d imu_acc = tail->acc;
            Eigen::Vector3d imu_gyro = tail->gyro;

            for (; it_pcl->curvature / double(1000) > head->offset; it_pcl--)
            {
                dt = it_pcl->curvature / double(1000) - head->offset;
                Eigen::Vector3d point(it_pcl->x, it_pcl->y, it_pcl->z);
                Eigen::Matrix3d point_rot = imu_rot * Sophus::SO3d::exp(imu_gyro * dt).matrix();
                Eigen::Vector3d point_pos = imu_pos + imu_vel * dt + 0.5 * imu_acc * dt * dt;
                Eigen::Vector3d p_compensate = cur_rot_ext.transpose() * (cur_rot.transpose() * (point_rot * (cur_rot_ext * point + cur_pos_ext) + point_pos - cur_pos) - cur_pos_ext);
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                if (it_pcl == package.cloud->points.begin())
                    break;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr LIOBuilder::lidarToWorld(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3, 3>(0, 0) = (kf.x().rot * kf.x().rot_ext).cast<float>();
        transform.block<3, 1>(0, 3) = (kf.x().rot * kf.x().pos_ext + kf.x().pos).cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_world, transform);
        return cloud_world;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr LIOBuilder::lidarToBody(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_body(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3, 3>(0, 0) = kf.x().rot_ext.cast<float>();
        transform.block<3, 1>(0, 3) = kf.x().pos_ext.cast<float>();
        pcl::transformPointCloud(*cloud, *cloud_body, transform);
        return cloud_body;
    }

    void LIOBuilder::process(SyncPackage &package)
    {
        if (status == LIOStatus::IMU_INIT)
        {
            if (initializeImu(package.imus))
            {
                status = LIOStatus::MAP_INIT;
                data_group.last_cloud_end_time = package.cloud_end_time;
            }
        }
        else if (status == LIOStatus::MAP_INIT)
        {
            undistortCloud(package);
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr point_world = lidarToWorld(package.cloud);
            status = LIOStatus::LIO_MAPPING;
        }
        else
        {
            undistortCloud(package);
            if (config.scan_resolution > 0.0)
            {
                scan_filter.setInputCloud(package.cloud);
                scan_filter.filter(*lidar_cloud);
            }
            else
            {
                pcl::copyPointCloud(*package.cloud, *lidar_cloud);
            }
        }
    }

}