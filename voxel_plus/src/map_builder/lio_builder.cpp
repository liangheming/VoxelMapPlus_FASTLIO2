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

        map = std::make_shared<VoxelMap>(config.voxel_size, config.max_layer, config.update_size_threshes, config.max_point_thresh, config.plane_thresh);

        lidar_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        kf.set_share_function(
            [this](kf::State &s, kf::SharedState &d)
            { sharedUpdateFunc(s, d); });
        data_group.residual_info.resize(10000);
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

            std::vector<PointWithCov> pv_list;
            Eigen::Matrix3d r_wl = kf.x().rot * kf.x().rot_ext;
            Eigen::Vector3d p_wl = kf.x().rot * kf.x().pos_ext + kf.x().pos;
            for (size_t i = 0; i < point_world->size(); i++)
            {
                PointWithCov pv;
                pv.point = Eigen::Vector3d(point_world->points[i].x, point_world->points[i].y, point_world->points[i].z);
                Eigen::Vector3d point_body(package.cloud->points[i].x, package.cloud->points[i].y, package.cloud->points[i].z);
                Eigen::Matrix3d point_cov;
                calcBodyCov(point_body, config.ranging_cov, config.angle_cov, point_cov);
                Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(point_body);

                point_cov = r_wl * point_cov * r_wl.transpose() +
                            point_crossmat * kf.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                            kf.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
                pv.cov = point_cov;
                pv_list.push_back(pv);
            }

            map->insert(pv_list);

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
            int size = lidar_cloud->size();
            for (int i = 0; i < size; i++)
            {
                data_group.residual_info[i].point_lidar = Eigen::Vector3d(lidar_cloud->points[i].x, lidar_cloud->points[i].y, lidar_cloud->points[i].z);
                calcBodyCov(data_group.residual_info[i].point_lidar, config.ranging_cov, config.angle_cov, data_group.residual_info[i].pcov);
            }
            kf.update();
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr point_world = lidarToWorld(lidar_cloud);
            std::vector<PointWithCov> pv_list;
            Eigen::Matrix3d r_wl = kf.x().rot * kf.x().rot_ext;
            Eigen::Vector3d p_wl = kf.x().rot * kf.x().pos_ext + kf.x().pos;
            for (int i = 0; i < size; i++)
            {
                PointWithCov pv;
                pv.point = Eigen::Vector3d(point_world->points[i].x, point_world->points[i].y, point_world->points[i].z);
                Eigen::Matrix3d cov = data_group.residual_info[i].pcov;
                Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(data_group.residual_info[i].point_lidar);
                pv.cov = r_wl * cov * r_wl.transpose() +
                         point_crossmat * kf.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                         kf.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
                pv_list.push_back(pv);
            }
            map->insert(pv_list);
        }
    }

    void LIOBuilder::sharedUpdateFunc(kf::State &state, kf::SharedState &shared_state)
    {
        Eigen::Matrix3d r_wl = state.rot * state.rot_ext;
        Eigen::Vector3d p_wl = state.rot * state.pos_ext + state.pos;

        int size = lidar_cloud->size();

#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < size; i++)
        {
            data_group.residual_info[i].is_valid = false;
            data_group.residual_info[i].current_layer = 0;
            data_group.residual_info[i].from_near = false;
            data_group.residual_info[i].point_world = r_wl * data_group.residual_info[i].point_lidar + p_wl;

            Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(data_group.residual_info[i].point_lidar);
            data_group.residual_info[i].cov = r_wl * data_group.residual_info[i].pcov * r_wl.transpose() +
                                              point_crossmat * kf.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                                              kf.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
            VoxelKey position = map->index(data_group.residual_info[i].point_world);
            auto iter = map->feat_map.find(position);
            if (iter != map->feat_map.end())
            {
                map->buildResidual(data_group.residual_info[i], iter->second.tree);
            }
        }

        shared_state.H.setZero();
        shared_state.b.setZero();
        Eigen::Matrix<double, 1, 12> J;
        Eigen::Matrix<double, 1, 6> J_v;
        int effect_num = 0;

        for (int i = 0; i < size; i++)
        {
            if (!data_group.residual_info[i].is_valid)
                continue;
            effect_num++;
            J.setZero();
            J_v.block<1, 3>(0, 0) = (data_group.residual_info[i].point_world - data_group.residual_info[i].plane_center).transpose();
            J_v.block<1, 3>(0, 3) = -data_group.residual_info[i].plane_norm.transpose();
            double r_cov = J_v * data_group.residual_info[i].plane_cov * J_v.transpose();
            r_cov += data_group.residual_info[i].plane_norm.transpose() * r_wl * data_group.residual_info[i].pcov * r_wl.transpose() * data_group.residual_info[i].plane_norm;
            // std::cout << 1.0 / r_cov << std::endl;
            double r_info = r_cov < 0.001 ? 1000 : 1 / r_cov;
            // double r_info = 1.0 / r_cov;
            assert(r_cov > 0.0);
            J.block<1, 3>(0, 0) = data_group.residual_info[i].plane_norm.transpose();
            J.block<1, 3>(0, 3) = -data_group.residual_info[i].plane_norm.transpose() * state.rot * Sophus::SO3d::hat(state.rot_ext * data_group.residual_info[i].point_lidar + state.pos_ext);
            if (config.estimate_ext)
            {
                J.block<1, 3>(0, 6) = -data_group.residual_info[i].plane_norm.transpose() * r_wl * Sophus::SO3d::hat(data_group.residual_info[i].point_lidar);
                J.block<1, 3>(0, 9) = data_group.residual_info[i].plane_norm.transpose() * state.rot;
            }
            shared_state.H += J.transpose() * r_info * J;
            shared_state.b += J.transpose() * r_info * data_group.residual_info[i].residual;
        }
        if (effect_num < 1)
            std::cout << "NO EFFECTIVE POINT";
        // std::cout << "=============== " << effect_num << " ===== "<<size<<" ======" << std::endl;
    }

}