#include "descriptor.h"

namespace std_desc
{
    void voxelFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr in, double voxel_size)
    {
        std::unordered_map<VoxelKey, MPoint, VoxelKey::Hasher> voxels;
        for (auto p : in->points)
        {
            VoxelKey k = VoxelKey::index(p.x, p.y, p.z, voxel_size);
            if (voxels.find(k) == voxels.end())
            {
                voxels[k].xyzi.setZero();
                voxels[k].count = 0;
            }
            voxels[k].xyzi(0) += p.x;
            voxels[k].xyzi(1) += p.y;
            voxels[k].xyzi(2) += p.z;
            voxels[k].xyzi(3) += p.intensity;
            voxels[k].count += 1;
        }
        in->clear();
        // std::vector<VoxelKey> keys;
        // for (auto iter = voxels.begin(); iter != voxels.end(); iter++)
        // {
        //     keys.push_back(iter->first);
        // }
        // std::sort(keys.begin(), keys.end(), [](VoxelKey &k1, VoxelKey &k2)
        //           {
        //     std::string ks1 =  std::to_string(k1.x) + std::to_string(k1.y)+std::to_string(k1.z);
        //     std::string ks2 =  std::to_string(k2.x) + std::to_string(k2.y)+std::to_string(k2.z);
        //     return ks1 > ks2; });
        for (auto it = voxels.begin(); it != voxels.end(); it++)
        // for (VoxelKey &ck_ : keys)
        {
            // auto it = voxels.find(ck_);
            pcl::PointXYZI p;
            p.x = it->second.xyzi(0) / static_cast<double>(it->second.count);
            p.y = it->second.xyzi(1) / static_cast<double>(it->second.count);
            p.z = it->second.xyzi(2) / static_cast<double>(it->second.count);
            p.intensity = it->second.xyzi(3) / static_cast<double>(it->second.count);
            in->push_back(p);
        }
    }

    Eigen::Matrix3d skew(const Eigen::Vector3d &vec)
    {
        Eigen::Matrix3d ret;
        ret << 0.0, -vec(2), vec(1),
            vec(2), 0.0, -vec(0),
            -vec(1), vec(0), 0.0;
        return ret;
    }

    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    std::vector<VoxelKey> VoxelKey::rangeNear(VoxelKey center, int range, bool exclude)
    {
        assert(range > 0);
        int len = range * 2 + 1;
        int size = exclude ? len * len * len - 1 : len * len * len;
        std::vector<VoxelKey> keys;
        keys.reserve(size);
        for (int i = -range; i <= range; i++)
            for (int j = -range; j <= range; j++)
                for (int k = -range; k <= range; k++)
                {
                    if (i == 0 && j == 0 && k == 0 && exclude)
                        continue;
                    keys.emplace_back(center.x + i, center.y + j, center.z + k);
                }
        return keys;
    }

    uint64_t STDManager::frame_count = 0;

    void STDManager::buildVoxels(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
    {
        for (auto &p : cloud->points)
        {
            VoxelKey k = VoxelKey::index(p.x, p.y, p.z, config.voxel_size);
            if (temp_voxels.find(k) == temp_voxels.end())
            {
                temp_voxels[k].cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
                temp_voxels[k].is_plane = false;
                temp_voxels[k].is_valid = false;
                temp_voxels[k].lamdas.setZero();
                temp_voxels[k].norms.setZero();
                temp_voxels[k].sum.setZero();
                temp_voxels[k].ppt.setZero();
            }
            temp_voxels[k].cloud->push_back(p);
            Eigen::Vector3d point_vec(p.x, p.y, p.z);
            temp_voxels[k].sum += point_vec;
            temp_voxels[k].ppt += point_vec * point_vec.transpose();
        }

        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            if (it->second.cloud->size() <= config.voxel_min_point)
            {
                it->second.is_valid = false;
                continue;
            }
            Eigen::Vector3d mean = it->second.sum / static_cast<double>(it->second.cloud->size());
            Eigen::Matrix3d cov = it->second.ppt / static_cast<double>(it->second.cloud->size()) - mean * mean.transpose();
            Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
            Eigen::Matrix3d norms = es.eigenvectors().real();
            Eigen::Vector3d lamdas = es.eigenvalues().real();
            Eigen::Matrix3d::Index evalsMin, evalsMax;
            lamdas.minCoeff(&evalsMin);
            lamdas.maxCoeff(&evalsMax);
            int evalsMid = 3 - evalsMin - evalsMax;
            it->second.mean = mean;
            if (lamdas(evalsMin) < config.voxel_plane_thresh)
            {
                it->second.is_plane = true;
                it->second.lamdas.x() = lamdas(evalsMin);
                it->second.lamdas.y() = lamdas(evalsMid);
                it->second.lamdas.z() = lamdas(evalsMax);
                it->second.norms.col(0) = norms.col(evalsMin);
                it->second.norms.col(1) = norms.col(evalsMid);
                it->second.norms.col(2) = norms.col(evalsMax);
                it->second.norm = it->second.norms.col(0);
                continue;
            }
            it->second.is_plane = false;
        }
    }

    void STDManager::buildConnections()
    {
        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            // 如果不是平面，则跳过
            if (!it->second.is_plane)
                continue;

            VoxelNode &c_node = it->second;
            for (int i = 0; i < 6; i++)
            {
                VoxelKey nk = it->first;
                if (i == 0)
                    nk.x = nk.x + 1;
                else if (i == 1)
                    nk.y = nk.y + 1;
                else if (i == 2)
                    nk.z = nk.z + 1;
                else if (i == 3)
                    nk.x = nk.x - 1;
                else if (i == 4)
                    nk.y = nk.y - 1;
                else if (i == 5)
                    nk.z = nk.z - 1;
                auto near = temp_voxels.find(nk);
                if (near == temp_voxels.end())
                {
                    c_node.connect_check[i] = true;
                    c_node.connect[i] = false;
                }
                else
                {
                    // 如果在其他节点被访问过，则跳过
                    if (c_node.connect_check[i])
                        continue;

                    VoxelNode &n_node = near->second;
                    c_node.connect_check[i] = true;
                    int j = i >= 3 ? i - 3 : i + 3;
                    n_node.connect_check[j] = true;
                    // 如果邻接节点是平面
                    if (n_node.is_plane)
                    {

                        Eigen::Vector3d normal_dif = c_node.norm - n_node.norm;
                        Eigen::Vector3d normal_add = c_node.norm + n_node.norm;
                        if (normal_add.norm() < config.norm_merge_thresh || normal_dif.norm() < config.norm_merge_thresh)
                        {
                            c_node.connect[i] = true;
                            n_node.connect[j] = true;
                            c_node.connect_nodes[i] = &n_node;
                            n_node.connect_nodes[j] = &c_node;
                        }
                        else
                        {
                            c_node.connect[i] = false;
                            n_node.connect[j] = false;
                        }
                    }
                    else
                    {
                        c_node.connect[i] = false;
                        n_node.connect[j] = true;
                        n_node.connect_nodes[j] = &c_node;
                    }
                }
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::nms2d(const Eigen::Vector3d &mean, const Eigen::Vector3d &norm, std::vector<Eigen::Vector3d> &proj_points)
    {
        // std::cout << mean.transpose() << ":" << norm.transpose() << std::endl;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr ret(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Vector3d x_axis(1, 1, 0);

        double A = norm(0), B = norm(1), C = norm(2);
        if (C != 0)
            x_axis[2] = -(A + B) / C;
        else if (B != 0)
            x_axis[1] = -A / B;
        else
        {
            x_axis[0] = 0;
            x_axis[1] = 1;
        }
        x_axis.normalize();
        Eigen::Vector3d y_axis = norm.cross(x_axis);
        std::vector<Eigen::Vector2d> point_list_2d;
        for (Eigen::Vector3d point : proj_points)
        {
            Eigen::Vector3d p2c = point - mean;
            double proj_x = p2c.dot(y_axis), proj_y = p2c.dot(x_axis), proj_z = p2c.dot(norm);
            if (std::abs(proj_z) < config.proj_min_dis || std::abs(proj_z) > config.proj_max_dis)
                continue;
            point_list_2d.push_back(Eigen::Vector2d(proj_x, proj_y));
        }
        if (point_list_2d.size() <= 5)
            return ret;
        double min_x = 10, max_x = -10, min_y = 10, max_y = -10;
        for (auto pi : point_list_2d)
        {
            if (pi[0] < min_x)
                min_x = pi[0];
            if (pi[0] > max_x)
                max_x = pi[0];
            if (pi[1] < min_y)
                min_y = pi[1];
            if (pi[1] > max_y)
                max_y = pi[1];
        }

        double segmen_len = config.nms_2d_range * config.proj_2d_resolution;
        int x_segment_num = (max_x - min_x) / segmen_len + 1;
        int y_segment_num = (max_y - min_y) / segmen_len + 1;
        int x_axis_len = (int)((max_x - min_x) / config.proj_2d_resolution + config.nms_2d_range);
        int y_axis_len = (int)((max_y - min_y) / config.proj_2d_resolution + config.nms_2d_range);

        double img_count_array[x_axis_len][y_axis_len];
        double mean_x_array[x_axis_len][y_axis_len];
        double mean_y_array[x_axis_len][y_axis_len];

        for (int x = 0; x < x_axis_len; x++)
        {
            for (int y = 0; y < y_axis_len; y++)
            {
                img_count_array[x][y] = 0;
                mean_x_array[x][y] = 0;
                mean_y_array[x][y] = 0;
            }
        }

        for (size_t i = 0; i < point_list_2d.size(); i++)
        {

            int x_index = (int)((point_list_2d[i][0] - min_x) / config.proj_2d_resolution);
            int y_index = (int)((point_list_2d[i][1] - min_y) / config.proj_2d_resolution);
            mean_x_array[x_index][y_index] += point_list_2d[i][0];
            mean_y_array[x_index][y_index] += point_list_2d[i][1];
            img_count_array[x_index][y_index]++;
        }
        std::vector<int> max_gradient_vec;
        std::vector<int> max_gradient_x_index_vec;
        std::vector<int> max_gradient_y_index_vec;

        for (int x_segment_index = 0; x_segment_index < x_segment_num; x_segment_index++)
        {
            for (int y_segment_index = 0; y_segment_index < y_segment_num; y_segment_index++)
            {
                double max_gradient = 0;
                int max_gradient_x_index = -10;
                int max_gradient_y_index = -10;
                for (int x_index = x_segment_index * config.nms_2d_range; x_index < (x_segment_index + 1) * config.nms_2d_range; x_index++)
                {
                    for (int y_index = y_segment_index * config.nms_2d_range; y_index < (y_segment_index + 1) * config.nms_2d_range; y_index++)
                    {
                        if (img_count_array[x_index][y_index] > max_gradient)
                        {
                            max_gradient = img_count_array[x_index][y_index];
                            max_gradient_x_index = x_index;
                            max_gradient_y_index = y_index;
                        }
                    }
                }
                if (max_gradient >= config.corner_thresh)
                {
                    max_gradient_vec.push_back(max_gradient);
                    max_gradient_x_index_vec.push_back(max_gradient_x_index);
                    max_gradient_y_index_vec.push_back(max_gradient_y_index);
                }
            }
        }

        for (int i = 0; i < max_gradient_vec.size(); i++)
        {
            double px = mean_x_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                        img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
            double py = mean_y_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                        img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
            Eigen::Vector3d coord = py * x_axis + px * y_axis + mean;
            pcl::PointXYZINormal pi;
            pi.x = coord[0];
            pi.y = coord[1];
            pi.z = coord[2];
            pi.intensity = max_gradient_vec[i];
            pi.curvature = 1.0;
            pi.normal_x = norm[0];
            pi.normal_y = norm[1];
            pi.normal_z = norm[2];
            ret->points.push_back(pi);
        }
        return ret;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::extractCorners()
    {
        std::vector<VoxelKey> keys;
        for (auto iter = temp_voxels.begin(); iter != temp_voxels.end(); iter++)
        {
            keys.push_back(iter->first);
        }
        std::sort(keys.begin(), keys.end(), [](VoxelKey &k1, VoxelKey &k2)
                  {
            std::string ks1 =  std::to_string(k1.x) + std::to_string(k1.y)+std::to_string(k1.z);
            std::string ks2 =  std::to_string(k2.x) + std::to_string(k2.y)+std::to_string(k2.z);
            return ks1 > ks2; });
        // TODO: 遍历的一致性约束

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_corner_points(new pcl::PointCloud<pcl::PointXYZINormal>);

        // for (auto iter = connect_voxels.begin(); iter != connect_voxels.end(); iter++)
        for (VoxelKey &ck_ : keys)
        {
            auto iter = temp_voxels.find(ck_);
            if (iter->second.is_plane)
                continue;

            VoxelKey ck = iter->first;
            VoxelNode *c_node = &iter->second;
            int connect_index = -1;
            for (int i = 0; i < 6; i++)
            {
                if (!c_node->connect[i])
                    continue;
                connect_index = i;
                VoxelNode *n_node = c_node->connect_nodes[connect_index];
                bool valid_plane = false;
                for (int j = 0; j < 6; j++)
                    if (n_node->connect_check[j])
                        if (n_node->connect[j])
                            valid_plane = true;

                if (!valid_plane)
                    continue;
                if (c_node->cloud->size() <= 10)
                    continue;
                Eigen::Vector3d projection_normal = n_node->norm;
                Eigen::Vector3d projection_center = n_node->mean;

                std::vector<Eigen::Vector3d> proj_points;

                for (VoxelKey nk : VoxelKey::rangeNear(ck, 1, false))
                {
                    auto iter_near = temp_voxels.find(nk);
                    if (iter_near == temp_voxels.end())
                        continue;
                    if (iter_near->second.is_plane)
                        continue;

                    bool skip = false;
                    if (iter_near->second.is_projected)
                    {
                        for (auto pnorm : iter_near->second.projected_norms)
                        {
                            Eigen::Vector3d normal_dif = projection_normal - pnorm;
                            Eigen::Vector3d normal_add = projection_normal + pnorm;
                            if (normal_dif.norm() < 0.5 || normal_add.norm() < 0.5)
                                skip = true;
                        }
                    }
                    if (skip)
                        continue;
                    iter_near->second.is_projected = true;
                    iter_near->second.projected_norms.push_back(projection_normal);
                    for (auto point : iter_near->second.cloud->points)
                    {
                        proj_points.emplace_back(point.x, point.y, point.z);
                    }
                }

                if (proj_points.size() == 0)
                    continue;
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr corners = nms2d(projection_center, projection_normal, proj_points);

                prepare_corner_points->points.insert(prepare_corner_points->points.begin(), corners->points.begin(), corners->points.end());
            }
        }

        if (prepare_corner_points->size() > 0)
            nms3d(prepare_corner_points);
        if (prepare_corner_points->size() > 0)
            // std::cout << "after nms3d: " << prepare_corner_points->size() << std::endl;

        if (prepare_corner_points->size() > config.max_corner_num)
        {
            std::sort(prepare_corner_points->points.begin(), prepare_corner_points->points.end(), [](pcl::PointXYZINormal &p1, pcl::PointXYZINormal &p2) -> bool
                      { return p1.intensity > p2.intensity; });
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
            prepare_key_cloud->points.assign(prepare_corner_points->points.begin(), prepare_corner_points->points.begin() + config.max_corner_num);
            prepare_corner_points = prepare_key_cloud;
        }
        return prepare_corner_points;
    }

    void STDManager::nms3d(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
        kd_tree.setInputCloud(prepare_key_cloud);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        // std::cout << "nms: " << prepare_key_cloud->size() << " range: " << config.nms_3d_range << std::endl;

        for (size_t i = 0; i < prepare_key_cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
            if (kd_tree.radiusSearch(searchPoint, config.nms_3d_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
            {
                for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
                {
                    if (pointIdxRadiusSearch[j] == i)
                        continue;
                    if (prepare_key_cloud->points[i].intensity <= prepare_key_cloud->points[pointIdxRadiusSearch[j]].intensity)
                    {
                        prepare_key_cloud->points[i].curvature = 0.0;
                        break;
                    }
                }
            }
        }
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZINormal>);
        temp->reserve(prepare_key_cloud->size());
        for (auto &p : prepare_key_cloud->points)
            if (p.curvature == 1.0)
                temp->push_back(p);
        prepare_key_cloud->clear();
        pcl::copyPointCloud(*temp, *prepare_key_cloud);
    }

    void STDManager::buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc)
    {
        std::unordered_set<VoxelKey, VoxelKey::Hasher> flag;
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
            new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(cloud);
        std::vector<int> pointIdxNKNSearch(config.desc_search_range);
        std::vector<float> pointNKNSquaredDistance(config.desc_search_range);
        for (size_t i = 0; i < cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = cloud->points[i];
            if (kd_tree->nearestKSearch(searchPoint, config.desc_search_range, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (int m = 1; m < config.desc_search_range - 1; m++)
                {
                    for (int n = m + 1; n < config.desc_search_range; n++)
                    {
                        pcl::PointXYZINormal p1 = searchPoint;
                        pcl::PointXYZINormal p2 = cloud->points[pointIdxNKNSearch[m]];
                        pcl::PointXYZINormal p3 = cloud->points[pointIdxNKNSearch[n]];

                        double s1 = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                        double s2 = sqrt(pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2));
                        double s3 = sqrt(pow(p3.x - p1.x, 2) + pow(p3.y - p1.y, 2) + pow(p3.z - p1.z, 2));

                        if (s1 < config.min_side_len || s1 > config.max_side_len ||
                            s2 < config.min_side_len || s2 > config.max_side_len ||
                            s3 < config.min_side_len || s3 > config.max_side_len)
                        {
                            continue;
                        }
                        std::vector<std::pair<double, int>> arr_pair{{s1, 0}, {s2, 1}, {s3, 2}};
                        std::sort(arr_pair.begin(), arr_pair.end(), [](std::pair<double, int> &i1, std::pair<double, int> &i2)
                                  { return i1.first < i2.first; });
                        std::vector<std::pair<double, pcl::PointXYZINormal>> temp_ps{{s1, p3}, {s2, p1}, {s3, p2}};
                        p1 = temp_ps[arr_pair[0].second].second;
                        p2 = temp_ps[arr_pair[1].second].second;
                        p3 = temp_ps[arr_pair[2].second].second;

                        s1 = temp_ps[arr_pair[0].second].first;
                        s2 = temp_ps[arr_pair[1].second].first;
                        s3 = temp_ps[arr_pair[2].second].first;
                        VoxelKey k = VoxelKey::index(s1, s2, s3, 0.001);
                        if (flag.find(k) != flag.end())
                        {
                            continue;
                        }
                        flag.insert(k);
                        STDDescriptor one_des;
                        one_des.center = Eigen::Vector3d(p1.x + p2.x + p3.x, p1.y + p2.y + p3.y, p1.z + p2.z + p3.z) / 3.0;
                        one_des.vertex_a = Eigen::Vector3d(p1.x, p1.y, p1.z);
                        one_des.vertex_b = Eigen::Vector3d(p2.x, p2.y, p2.z);
                        one_des.vertex_c = Eigen::Vector3d(p3.x, p3.y, p3.z);
                        one_des.attached = Eigen::Vector3d(p1.intensity, p2.intensity, p3.intensity);
                        one_des.side_length = Eigen::Vector3d(s1, s2, s3);
                        one_des.norms.col(0) = Eigen::Vector3d(p1.normal_x, p1.normal_y, p1.normal_z);
                        one_des.norms.col(1) = Eigen::Vector3d(p2.normal_x, p2.normal_y, p2.normal_z);
                        one_des.norms.col(2) = Eigen::Vector3d(p3.normal_x, p3.normal_y, p3.normal_z);
                        one_des.angle = Eigen::Vector3d(
                            std::abs(5 * one_des.norms.col(0).dot(one_des.norms.col(1))),
                            std::abs(5 * one_des.norms.col(1).dot(one_des.norms.col(2))),
                            std::abs(5 * one_des.norms.col(2).dot(one_des.norms.col(0))));
                        desc.push_back(one_des);
                    }
                }
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::currentPlaneCloud()
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr ret(new pcl::PointCloud<pcl::PointXYZINormal>);
        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            if (!it->second.is_plane)
                continue;
            pcl::PointXYZINormal p;
            p.x = it->second.mean.x();
            p.y = it->second.mean.y();
            p.z = it->second.mean.z();
            p.normal_x = it->second.norms.col(0).x();
            p.normal_y = it->second.norms.col(0).y();
            p.normal_z = it->second.norms.col(0).z();
            ret->push_back(p);
        }
        return ret;
    }

    STDFeature STDManager::extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
    {
        std::unordered_map<VoxelKey, VoxelNode, VoxelKey::Hasher>().swap(temp_voxels);

        buildVoxels(cloud);

        buildConnections();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr corners = extractCorners();

        STDFeature feature;

        if (corners->size() > 0)
            buildDescriptor(corners, feature.descs);

        feature.cloud = currentPlaneCloud();

        feature.id = frame_count;
        return feature;
    }

    void STDManager::insert(STDFeature &feature)
    {
        cloud_vec.push_back(feature.cloud);
        feature.id = frame_count;
        for (STDDescriptor &des : feature.descs)
        {
            des.id = frame_count;
            VoxelKey position;
            position.x = (int)(des.side_length[0] / config.side_resolution + 0.5);
            position.y = (int)(des.side_length[1] / config.side_resolution + 0.5);
            position.z = (int)(des.side_length[2] / config.side_resolution + 0.5);
            auto iter = data_base.find(position);
            if (iter == data_base.end())
            {
                data_base[position] = std::vector<STDDescriptor>();
            }
            data_base[position].push_back(des);
        }
        frame_count++;
    }

    void STDManager::triangleSolver(std::pair<STDDescriptor, STDDescriptor> &pair, Eigen::Matrix3d &rot, Eigen::Vector3d &trans)
    {
        Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
        src.col(0) = pair.first.vertex_a - pair.first.center;
        src.col(1) = pair.first.vertex_b - pair.first.center;
        src.col(2) = pair.first.vertex_c - pair.first.center;
        ref.col(0) = pair.second.vertex_a - pair.second.center;
        ref.col(1) = pair.second.vertex_b - pair.second.center;
        ref.col(2) = pair.second.vertex_c - pair.second.center;
        Eigen::Matrix3d covariance = src * ref.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d U = svd.matrixU();
        rot = V * U.transpose();
        if (rot.determinant() < 0)
        {
            Eigen::Matrix3d K;
            K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
            rot = V * K * U.transpose();
        }
        trans = -rot * pair.first.center + pair.second.center;
    }

    void STDManager::selectCanidates(STDFeature &feature, std::vector<STDMatch> &match_list)
    {
        std::vector<Eigen::Vector3i> voxel_round;

        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    Eigen::Vector3i voxel_inc(x, y, z);
                    voxel_round.push_back(voxel_inc);
                }
            }
        }

        std::vector<bool> useful_match(feature.descs.size(), false);
        std::vector<std::vector<size_t>> useful_match_index(feature.descs.size());
        std::vector<std::vector<VoxelKey>> useful_match_position(feature.descs.size());
        // 根据三角边长选择候选
        for (size_t i = 0; i < feature.descs.size(); i++)
        {
            STDDescriptor &src_std = feature.descs[i];
            VoxelKey position;
            double dis_threshold = (src_std.side_length / config.side_resolution).norm() * config.rough_dis_threshold;

            for (Eigen::Vector3i &voxel_inc : voxel_round)
            {
                position.x = (int)(src_std.side_length[0] / config.side_resolution + voxel_inc[0]);
                position.y = (int)(src_std.side_length[1] / config.side_resolution + voxel_inc[1]);
                position.z = (int)(src_std.side_length[2] / config.side_resolution + voxel_inc[2]);
                Eigen::Vector3d voxel_center((double)position.x + 0.5, (double)position.y + 0.5, (double)position.z + 0.5);

                if ((src_std.side_length / config.side_resolution - voxel_center).norm() >= 1.5)
                    continue;
                auto iter = data_base.find(position);
                if (iter == data_base.end())
                    continue;

                for (size_t j = 0; j < data_base[position].size(); j++)
                {
                    if ((feature.id - data_base[position][j].id) <= config.skip_near_num)
                        continue;
                    double dis = (src_std.side_length / config.side_resolution - data_base[position][j].side_length / config.side_resolution).norm();
                    if (dis >= dis_threshold)
                        continue;
                    double vertex_attach_diff = 2.0 * (src_std.attached - data_base[position][j].attached).norm() /
                                                (src_std.attached + data_base[position][j].attached).norm();
                    if (vertex_attach_diff >= config.vertex_diff_threshold)
                        continue;
                    useful_match[i] = true;
                    useful_match_position[i].push_back(position);
                    useful_match_index[i].push_back(j);
                }
            }
        }

        std::unordered_map<uint64_t, int> match_arr;
        std::vector<int> match_id_vec;
        std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder;

        // 记录所有匹配上的frame_id
        for (size_t i = 0; i < useful_match.size(); i++)
        {
            if (!useful_match[i])
                continue;
            for (size_t j = 0; j < useful_match_index[i].size(); j++)
            {
                uint64_t match_id = data_base[useful_match_position[i][j]][useful_match_index[i][j]].id;
                if (match_arr.find(match_id) == match_arr.end())
                    match_arr[match_id] = 0;
                match_arr[match_id]++;
                index_recorder.emplace_back(i, j);
                match_id_vec.push_back(match_id);
            }
        }

        for (int cnt = 0; cnt < config.candidate_num; cnt++)
        {

            int max_vote = 1;
            uint64_t max_vote_index = -1;

            for (auto it = match_arr.begin(); it != match_arr.end(); it++)
            {
                if (it->second > max_vote)
                {
                    max_vote = it->second;
                    max_vote_index = it->first;
                }
            }
            if (max_vote_index < 0 || max_vote < 5)
                break;
            STDMatch match_triangle;
            match_arr[max_vote_index] = 0;
            match_triangle.match_id = max_vote_index;
            for (size_t i = 0; i < index_recorder.size(); i++)
            {
                if (match_id_vec[i] != max_vote_index)
                    continue;
                std::pair<STDDescriptor, STDDescriptor> single_match_pair;
                single_match_pair.first = feature.descs[index_recorder[i][0]];
                single_match_pair.second =
                    data_base[useful_match_position[index_recorder[i][0]][index_recorder[i][1]]]
                             [useful_match_index[index_recorder[i][0]][index_recorder[i][1]]];
                match_triangle.match_pairs.push_back(single_match_pair);
            }
            match_list.push_back(match_triangle);
        }
    }

    double STDManager::verifyCandidate(STDMatch &matches,
                                       Eigen::Matrix3d &rot,
                                       Eigen::Vector3d &trans,
                                       std::vector<std::pair<STDDescriptor, STDDescriptor>> &success_match_vec,
                                       pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud)
    {
        success_match_vec.clear();
        int skip_len = (int)(matches.match_pairs.size() / 50) + 1;
        int use_size = matches.match_pairs.size() / skip_len;
        std::vector<int> vote_list(use_size);
        for (size_t i = 0; i < use_size; i++)
        {
            auto single_pair = matches.match_pairs[i * skip_len];
            Eigen::Matrix3d test_rot;
            Eigen::Vector3d test_t;
            int vote = 0;
            triangleSolver(single_pair, test_rot, test_t);
            for (size_t j = 0; j < matches.match_pairs.size(); j++)
            {
                auto verify_pair = matches.match_pairs[j];
                Eigen::Vector3d A = verify_pair.first.vertex_a;
                Eigen::Vector3d A_transform = test_rot * A + test_t;
                Eigen::Vector3d B = verify_pair.first.vertex_b;
                Eigen::Vector3d B_transform = test_rot * B + test_t;
                Eigen::Vector3d C = verify_pair.first.vertex_c;
                Eigen::Vector3d C_transform = test_rot * C + test_t;
                double dis_A = (A_transform - verify_pair.second.vertex_a).norm();
                double dis_B = (B_transform - verify_pair.second.vertex_b).norm();
                double dis_C = (C_transform - verify_pair.second.vertex_c).norm();
                if (dis_A < config.verify_dis_thresh && dis_B < config.verify_dis_thresh && dis_C < config.verify_dis_thresh)
                    vote++;
            }
            vote_list[i] = vote;
        }

        int max_vote_index = 0;
        int max_vote = 0;
        for (size_t i = 0; i < vote_list.size(); i++)
        {
            if (max_vote < vote_list[i])
            {
                max_vote_index = i;
                max_vote = vote_list[i];
            }
        }

        // std::cout << "max vot" << max_vote << std::endl;

        if (max_vote < 4) // 4
            return -1.0;
        auto best_pair = matches.match_pairs[max_vote_index * skip_len];
        triangleSolver(best_pair, rot, trans);
        for (size_t j = 0; j < matches.match_pairs.size(); j++)
        {
            auto verify_pair = matches.match_pairs[j];
            Eigen::Vector3d A = verify_pair.first.vertex_a;
            Eigen::Vector3d A_transform = rot * A + trans;
            Eigen::Vector3d B = verify_pair.first.vertex_b;
            Eigen::Vector3d B_transform = rot * B + trans;
            Eigen::Vector3d C = verify_pair.first.vertex_c;
            Eigen::Vector3d C_transform = rot * C + trans;
            double dis_A = (A_transform - verify_pair.second.vertex_a).norm();
            double dis_B = (B_transform - verify_pair.second.vertex_b).norm();
            double dis_C = (C_transform - verify_pair.second.vertex_c).norm();
            if (dis_A < config.verify_dis_thresh && dis_B < config.verify_dis_thresh && dis_C < config.verify_dis_thresh)
                success_match_vec.push_back(verify_pair);
        }
        // std::cout << "do verify" << std::endl;

        return verifyGeoPlane(plane_cloud, cloud_vec[matches.match_id], rot, trans);
    }

    double STDManager::verifyGeoPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                      const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(target_cloud);
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        double useful_match = 0;
        for (size_t i = 0; i < source_cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = source_cloud->points[i];
            Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
            pi = rot * pi + trans;
            Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
            ni = rot * ni;
            pcl::PointXYZINormal use_search_point;
            use_search_point.x = pi.x();
            use_search_point.y = pi.y();
            use_search_point.z = pi.z();
            if (kd_tree->nearestKSearch(use_search_point, 3, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[j]];
                    Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
                    Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
                    Eigen::Vector3d normal_inc = ni - tni;
                    Eigen::Vector3d normal_add = ni + tni;
                    double point_to_plane = std::abs(tni.transpose() * (pi - tpi));

                    // std::cout << "point_to_plane: " << point_to_plane << ":" << trans.transpose() << std::endl;

                    if ((normal_inc.norm() < config.norm_merge_thresh || normal_add.norm() < config.norm_merge_thresh) && point_to_plane < config.geo_verify_dis_thresh)
                    {
                        useful_match++;
                        break;
                    }
                }
            }
        }
        return useful_match / source_cloud->size();
    }

    double STDManager::verifyGeoPlaneICP(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                         pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                         Eigen::Matrix3d &rot, Eigen::Vector3d &trans)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(target_cloud);
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        std::vector<Eigen::Vector3d> matched_source;
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched_target;

        for (size_t i = 0; i < source_cloud->size(); i++)
        {
            pcl::PointXYZINormal &searchPoint = source_cloud->points[i];
            Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);

            pi = rot * pi + trans;

            Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
            ni = rot * ni;

            pcl::PointXYZINormal use_search_point;
            use_search_point.x = pi[0];
            use_search_point.y = pi[1];
            use_search_point.z = pi[2];
            if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[0]];
                Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
                Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
                Eigen::Vector3d normal_inc = ni - tni;
                Eigen::Vector3d normal_add = ni + tni;
                double point_to_point_dis = (pi - tpi).norm();
                double point_to_plane = std::abs(tni.transpose() * (pi - tpi));
                if ((normal_inc.norm() < config.norm_merge_thresh ||
                     normal_add.norm() < config.norm_merge_thresh) &&
                    point_to_plane < config.geo_verify_dis_thresh && point_to_point_dis < 3)
                {
                    matched_source.emplace_back(searchPoint.x, searchPoint.y, searchPoint.z);
                    matched_target.emplace_back(tpi, tni);
                }
            }
        }

        Eigen::Matrix<double, 1, 6> J;
        Eigen::Matrix<double, 6, 6> H;
        Eigen::Matrix<double, 6, 1> b;
        Eigen::Matrix<double, 6, 1> delta;

        Eigen::Matrix3d opti_rot = rot;
        Eigen::Vector3d opti_trans = trans;
        double sum_res = -1.0;
        for (size_t i = 0; i < config.max_iter; i++)
        {
            H.setZero();
            b.setZero();
            sum_res = 0.0;
            for (size_t j = 0; j < matched_source.size(); j++)
            {
                Eigen::Vector3d &pi = matched_source[j];
                Eigen::Vector3d &tpi = matched_target[j].first;
                Eigen::Vector3d &tni = matched_target[j].second;
                J.block<1, 3>(0, 0) = -tni.transpose() * opti_rot * Sophus::SO3d::hat(pi);
                J.block<1, 3>(0, 3) = tni.transpose();
                double res = tni.dot(opti_rot * pi + opti_trans - tpi);
                H += J.transpose() * J;
                b -= J.transpose() * res;
                sum_res += std::abs(res);
            }
            delta = H.inverse() * b;
            opti_rot *= Sophus::SO3d::exp(delta.block<3, 1>(0, 0)).matrix();
            opti_trans += delta.block<3, 1>(3, 0);
            if (delta.maxCoeff() < config.iter_eps)
                break;
        }
        rot = opti_rot;
        trans = opti_trans;
        if (matched_source.size() > 0)
            return sum_res / static_cast<double>(matched_source.size());
        return sum_res;
    }

    LoopResult STDManager::searchLoop(STDFeature &feature)
    {
        LoopResult result;
        result.valid = false;
        result.match_id = -1;
        result.match_score = 0.0;
        if (feature.descs.size() == 0)
            return result;

        std::vector<STDMatch> candidate_matcher_vec;
        selectCanidates(feature, candidate_matcher_vec);

        for (size_t i = 0; i < candidate_matcher_vec.size(); i++)
        {
            Eigen::Vector3d trans;
            Eigen::Matrix3d rot;
            std::vector<std::pair<STDDescriptor, STDDescriptor>> sucess_match_vec;
            double verify_score = verifyCandidate(candidate_matcher_vec[i], rot, trans, sucess_match_vec, feature.cloud);

            // std::cout << "verify_score: " << verify_score << std::endl;

            if (verify_score > result.match_score)
            {
                result.match_score = verify_score;
                result.match_id = candidate_matcher_vec[i].match_id;
                result.translation = trans;
                result.rotation = rot;
                result.match_pairs = sucess_match_vec;
            }
        }
        if (result.match_score > config.icp_thresh)
            result.valid = true;
        return result;
    }
} // namespace name
