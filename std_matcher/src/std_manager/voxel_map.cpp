#include "voxel_map.h"

namespace stdes
{
    VoxelGrid::VoxelGrid(double _plane_thesh, int _min_num_thresh) : plane_thresh(_plane_thesh), min_num_thresh(_min_num_thresh)
    {
        sum.setZero();
        ppt.setZero();
        clouds.clear();
        is_plane = false;
    }

    void VoxelGrid::addPoint(const pcl::PointXYZINormal &p)
    {
        Eigen::Vector3d point(p.x, p.y, p.z);
        sum += point;
        ppt += point * point.transpose();
        clouds.push_back(p);
    }

    void VoxelGrid::buildPlane()
    {
        if (clouds.size() <= min_num_thresh)
        {
            is_plane = false;
            return;
        }
        Eigen::Vector3d mean = sum / static_cast<double>(clouds.size());
        Eigen::Matrix3d cov = ppt / static_cast<double>(clouds.size()) - mean * mean.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        norms = es.eigenvectors();
        lamdas = es.eigenvalues();
        if (lamdas(0) < plane_thresh)
            is_plane = true;
        else
            is_plane = false;
    }

    VoxelMap::VoxelMap(double _voxel_size, double _plane_thesh, int _min_num_thresh) : voxel_size(_voxel_size), plane_thresh(_plane_thesh), min_num_thresh(_min_num_thresh)
    {
        voxels.clear();
        plane_voxels.clear();
        planes.clear();
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> VoxelMap::coloredPlaneCloud(bool with_corner)
    {
        int size = planes.size();
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> ret;
        if (size > 0)
            ret.reserve(size);

        for (int i = 0; i < size; i++)
        {

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_visual(new pcl::PointCloud<pcl::PointXYZRGBA>);
            Plane &plane_i = planes[i];
            for (pcl::PointXYZINormal &point : plane_i.sur_cloud->points)
            {
                pcl::PointXYZRGBA sp;
                RGB rgb = valueToColor((double)i, 0.0, (double)size);
                sp.r = rgb.r;
                sp.b = rgb.b;
                sp.g = rgb.g;
                sp.x = point.x;
                sp.y = point.y;
                sp.z = point.z;
                cloud_visual->push_back(sp);
            }
            if (with_corner)
            {
                for (pcl::PointXYZINormal &point : plane_i.corner_cloud->points)
                {
                    pcl::PointXYZRGBA sp;
                    sp.r = 255;
                    sp.b = 255;
                    sp.g = 255;
                    sp.x = point.x;
                    sp.y = point.y;
                    sp.z = point.z;
                    cloud_visual->push_back(sp);
                }
            }

            ret.push_back(cloud_visual);
        }

        return ret;
    }

    VoxelKey VoxelMap::index(const pcl::PointXYZINormal &p)
    {
        Eigen::Vector3d point(p.x, p.y, p.z);
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    VoxelKey VoxelMap::index(double x, double y, double z, double resolution)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    std::vector<VoxelKey> VoxelMap::nears(const VoxelKey &center, int range)
    {
        std::vector<VoxelKey> ret;
        ret.reserve(range * range * range - 1);
        for (int i = -range; i <= range; i++)
        {
            for (int j = -range; j <= range; j++)
            {
                for (int k = -range; k <= range; k++)
                {
                    if (i == 0 && j == 0 && k == 0)
                        continue;
                    int64_t x = center.x + i, y = center.y + j, z = center.z + k;
                    ret.emplace_back(x, y, z);
                }
            }
        }
        return ret;
    }

    std::vector<VoxelKey> VoxelMap::sixNears(const VoxelKey &center)
    {
        std::vector<VoxelKey> ret;
        ret.reserve(6);
        ret.emplace_back(center.x - 1, center.y, center.z);
        ret.emplace_back(center.x, center.y - 1, center.z);
        ret.emplace_back(center.x, center.y, center.z - 1);
        ret.emplace_back(center.x + 1, center.y, center.z);
        ret.emplace_back(center.x, center.y + 1, center.z);
        ret.emplace_back(center.x, center.y, center.z + 1);
        return ret;
    }

    void VoxelMap::build(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        for (auto &p : cloud->points)
        {
            VoxelKey k = index(p);
            if (voxels.find(k) == voxels.end())
            {
                voxels[k] = std::make_shared<VoxelGrid>(plane_thresh, min_num_thresh);
            }
            voxels[k]->addPoint(p);
        }

        for (auto it = voxels.begin(); it != voxels.end(); it++)
        {
            it->second->buildPlane();
            if (it->second->is_plane)
            {
                plane_voxels.push_back(it->first);
            }
        }
    }

    void VoxelMap::mergePlanes()
    {
        std::unordered_map<VoxelKey, bool, VoxelKey::Hasher> flags;
        for (VoxelKey &k : plane_voxels)
            flags[k] = true;

        // 这里进行平面的合并，同时记录corner voxels;
        for (VoxelKey &k : plane_voxels)
        {
            if (!flags[k])
                continue;
            Plane p;
            p.sur_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
            p.corner_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
            p.corner_voxels.clear();
            std::queue<VoxelKey> buffer;
            buffer.push(k);

            while (!buffer.empty())
            {

                VoxelKey ck = buffer.front();
                buffer.pop();
                std::shared_ptr<VoxelGrid> cv = voxels[ck];
                flags[ck] = false;
                p.sum += cv->sum;
                p.ppt += cv->ppt;
                *p.sur_cloud += cv->clouds;
                p.num += 1;
                for (VoxelKey &nk : sixNears(ck))
                {
                    auto n_it = voxels.find(nk);
                    if (n_it == voxels.end())
                        continue;
                    if (n_it->second->is_plane)
                    {
                        if (((cv->norms.col(0) - n_it->second->norms.col(0)).norm() < merge_thresh || (cv->norms.col(0) + n_it->second->norms.col(0)).norm() < merge_thresh))
                        {
                            if (flags[n_it->first])
                                buffer.push(n_it->first);
                        }
                    }
                    else
                    {
                        // 如果不是平面则添加到cornner里面
                        if (n_it->second->clouds.size() > min_num_thresh)
                        {
                            if (p.corner_voxels.find(n_it->first) == p.corner_voxels.end())
                            {
                                *p.corner_cloud += n_it->second->clouds;
                                p.corner_voxels.insert(n_it->first);
                            }
                        }
                    }
                }
            }

            if (p.num <= 1)
                continue;
            std::unordered_set<VoxelKey, VoxelKey::Hasher> copy_corner_voxels(p.corner_voxels.begin(), p.corner_voxels.end());
            for (auto corner_keyset = copy_corner_voxels.begin(); corner_keyset != copy_corner_voxels.end(); corner_keyset++)
            {
                std::vector<VoxelKey> near_corner_ks = VoxelMap::nears(*corner_keyset, 1);
                for (VoxelKey &near_corner_k : near_corner_ks)
                {
                    auto near_corner_k_in_global_voxel = voxels.find(near_corner_k);
                    if (near_corner_k_in_global_voxel == voxels.end() || p.corner_voxels.find(near_corner_k) != p.corner_voxels.end() || near_corner_k_in_global_voxel->second->is_plane)
                        continue;
                    *p.corner_cloud += near_corner_k_in_global_voxel->second->clouds;
                    p.corner_voxels.insert(near_corner_k);
                }
            }

            if (p.corner_cloud->size() <= min_num_thresh)
                continue;

            p.mean = p.sum / static_cast<double>(p.sur_cloud->size());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(p.ppt / static_cast<double>(p.sur_cloud->size()) - p.mean * p.mean.transpose());
            p.norms = es.eigenvectors();
            p.lamdas = es.eigenvalues();
            if (p.lamdas(0) > plane_thresh)
                continue;
            planes.push_back(p);
        }
    }

    void VoxelMap::reset()
    {
        if (voxels.size() > 0)
            FeatMap().swap(voxels);
        if (plane_voxels.size() > 0)
            std::vector<VoxelKey>().swap(plane_voxels);
        if (planes.size() > 0)
            std::vector<Plane>().swap(planes);
    }

    STDExtractor::STDExtractor(std::shared_ptr<VoxelMap> _voxel_map) : voxel_map(_voxel_map)
    {
    }

    void STDExtractor::extract(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, uint64_t frame_id, std::vector<STDDescriptor> &descs)
    {
        voxel_map->reset();
        voxel_map->build(cloud);
        voxel_map->mergePlanes();
        descs.clear();
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);

        for (const Plane &plane : voxel_map->planes)
        {
            *prepare_key_cloud += *projectCornerNMS(plane);
        }

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr candidates = nms3D(prepare_key_cloud);

        std::cout << "corner size: " << candidates->size() << std::endl;

        if (candidates->points.size() > max_corner_num)
        {
            std::sort(candidates->points.begin(), candidates->points.end(), [](pcl::PointXYZINormal &p1, pcl::PointXYZINormal &p2) -> bool
                      { return p1.intensity > p2.intensity; });
            prepare_key_cloud->clear();
            prepare_key_cloud->points.assign(candidates->points.begin(), candidates->points.begin() + max_corner_num);
            candidates = prepare_key_cloud;
        }
        buildDescriptor(candidates, descs);
    }

    std::vector<VoxelKey> STDExtractor::nears2d(const VoxelKey &center, int range)
    {
        int len = range / 2;
        std::vector<VoxelKey> nears;
        nears.reserve((len * 2 + 1) * (len * 2 + 1) - 1);
        for (int i = -len; i <= len; i++)
        {
            for (int j = -len; j <= len; j++)
            {
                if (i == 0 && j == 0)
                    continue;
                nears.emplace_back(center.x + i, center.y + j, 0);
            }
        }
        return nears;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDExtractor::projectCornerNMS(const Plane &plane)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_point(new pcl::PointCloud<pcl::PointXYZINormal>);
        std::unordered_map<VoxelKey, Grid2d, VoxelKey::Hasher> image_grids;
        for (int i = 0; i < plane.corner_cloud->points.size(); i++)
        {
            pcl::PointXYZINormal p = plane.corner_cloud->points[i];
            Eigen::Vector3d dis_vec = Eigen::Vector3d(p.x, p.y, p.z) - plane.mean;
            Eigen::Vector3d plane_p_vec(dis_vec.dot(plane.norms.col(2)), dis_vec.dot(plane.norms.col(1)), dis_vec.dot(plane.norms.col(0)));
            Eigen::Vector3d idx = (plane_p_vec / image_resolution).array().floor();
            VoxelKey k(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), 0);
            if (plane_p_vec.z() <= project_min_dis || plane_p_vec.z() >= project_max_dis)
                continue;

            if (image_grids.find(k) == image_grids.end())
            {
                image_grids[k].mean = plane.mean;
                image_grids[k].norms = plane.norms;
                image_grids[k].xy = Eigen::Vector2d(plane_p_vec.x(), plane_p_vec.y());
                image_grids[k].num = 1;
                image_grids[k].flag = true;
            }
            else
            {
                image_grids[k].xy += Eigen::Vector2d(plane_p_vec.x(), plane_p_vec.y());
                image_grids[k].num += 1;
            }
        }

        // 在一定范围内进行非极大值抑制
        for (auto it = image_grids.begin(); it != image_grids.end(); it++)
        {
            VoxelKey ck = it->first;
            Grid2d &g2d = it->second;
            if (!g2d.flag)
                continue;
            for (VoxelKey &nk : nears2d(ck, nms_range))
            {
                if (image_grids.find(nk) == image_grids.end())
                    continue;
                if (g2d.num > image_grids[nk].num)
                    image_grids[nk].flag = false;
                else if (image_grids[nk].num > g2d.num)
                    g2d.flag = false;
            }
        }

        for (auto it = image_grids.begin(); it != image_grids.end(); it++)
        {
            if (!it->second.flag)
                continue;
            pcl::PointXYZINormal p;
            p.intensity = static_cast<float>(it->second.num);
            if (p.intensity < 10)
                continue;
            Eigen::Vector2d mean_xy = it->second.xy / static_cast<double>(it->second.num);
            Eigen::Vector3d corner_point_in_plane = mean_xy.x() * it->second.norms.col(2) + mean_xy.y() * it->second.norms.col(1) + it->second.mean;
            p.x = corner_point_in_plane.x();
            p.y = corner_point_in_plane.y();
            p.z = corner_point_in_plane.z();
            p.curvature = 1.0;
            p.normal_x = plane.norms.col(0).x();
            p.normal_y = plane.norms.col(0).y();
            p.normal_z = plane.norms.col(0).z();
            filtered_point->push_back(p);
        }

        return filtered_point;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDExtractor::nms3D(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
        kd_tree.setInputCloud(prepare_key_cloud);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        for (size_t i = 0; i < prepare_key_cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
            if (kd_tree.radiusSearch(searchPoint, nms_3d_range, pointIdxRadiusSearch,
                                     pointRadiusSquaredDistance) > 0)
            {
                for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
                {
                    if (pointIdxRadiusSearch[j] == i)
                    {
                        continue;
                    }
                    if (prepare_key_cloud->points[i].intensity <=
                        prepare_key_cloud->points[pointIdxRadiusSearch[j]].intensity)
                    {
                        prepare_key_cloud->points[i].curvature = 0.0;
                        break;
                    }
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr candidates(new pcl::PointCloud<pcl::PointXYZINormal>);
        for (pcl::PointXYZINormal &p : prepare_key_cloud->points)
        {
            if (p.curvature == 1.0)
                candidates->push_back(p);
        }

        return candidates;
    }

    void STDExtractor::buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc)
    {
        std::unordered_set<VoxelKey, VoxelKey::Hasher> flag;
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
            new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(cloud);
        // std::cout << cloud->size() << std::endl;
        std::vector<int> pointIdxNKNSearch(descriptor_near_num);
        std::vector<float> pointNKNSquaredDistance(descriptor_near_num);
        for (size_t i = 0; i < cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = cloud->points[i];
            if (kd_tree->nearestKSearch(searchPoint, descriptor_near_num, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (int m = 1; m < descriptor_near_num - 1; m++)
                {
                    for (int n = m + 1; n < descriptor_near_num; n++)
                    {
                        pcl::PointXYZINormal p1 = searchPoint;
                        pcl::PointXYZINormal p2 = cloud->points[pointIdxNKNSearch[m]];
                        pcl::PointXYZINormal p3 = cloud->points[pointIdxNKNSearch[n]];

                        double s1 = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                        double s2 = sqrt(pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2));
                        double s3 = sqrt(pow(p3.x - p1.x, 2) + pow(p3.y - p1.y, 2) + pow(p3.z - p1.z, 2));

                        if (s1 < min_dis_threshold || s1 > max_dis_threshold ||
                            s2 < min_dis_threshold || s2 > max_dis_threshold ||
                            s3 < min_dis_threshold || s3 > max_dis_threshold)
                        {
                            continue;
                        }
                        std::vector<std::pair<double, int>> arr_pair{{s1, 0}, {s2, 1}, {s3, 2}};
                        std::sort(arr_pair.begin(), arr_pair.end(), [](std::pair<double, int> &i1, std::pair<double, int> &i2)
                                  { return i1.first < i2.first; });
                        std::vector<std::pair<double, pcl::PointXYZINormal>> temp_ps{{s1, p1}, {s2, p2}, {s3, p3}};
                        p1 = temp_ps[arr_pair[0].second].second;
                        p2 = temp_ps[arr_pair[1].second].second;
                        p3 = temp_ps[arr_pair[2].second].second;

                        s1 = temp_ps[arr_pair[0].second].first;
                        s2 = temp_ps[arr_pair[1].second].first;
                        s3 = temp_ps[arr_pair[2].second].first;
                        VoxelKey k = VoxelMap::index(s1, s2, s3, 0.001);
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
                            std::abs(one_des.norms.col(0).dot(one_des.norms.col(1))),
                            std::abs(one_des.norms.col(1).dot(one_des.norms.col(2))),
                            std::abs(one_des.norms.col(2).dot(one_des.norms.col(0))));
                        desc.push_back(one_des);
                    }
                }
            }
        }
    }

} // namespace stdes
