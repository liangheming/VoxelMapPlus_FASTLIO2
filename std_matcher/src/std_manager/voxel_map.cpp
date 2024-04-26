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
        if (clouds.size() < min_num_thresh)
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
            // std::cout << "merged_plane_num: " << plane_i.num << " sur_cloud: " << plane_i.sur_cloud->size() << " cor_cloud: " << plane_i.corner_cloud->size() << std::endl;
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
                        else
                        {
                            if (p.corner_voxels.find(n_it->first) == p.corner_voxels.end())
                            {
                                *p.corner_cloud += n_it->second->clouds;
                                p.corner_voxels.insert(n_it->first);
                            }
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

            if (p.num > 1 && p.corner_cloud->size() > min_num_thresh)
            {
                p.mean = p.sum / static_cast<double>(p.sur_cloud->size());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(p.ppt / static_cast<double>(p.sur_cloud->size()) - p.mean * p.mean.transpose());
                p.norms = es.eigenvectors();
                p.lamdas = es.eigenvalues();
                if (p.lamdas(0) > plane_thresh)
                    continue;
                planes.push_back(p);
            }
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

    void STDExtractor::extract(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        voxel_map->reset();
        voxel_map->build(cloud);
        voxel_map->mergePlanes();
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        for (const Plane &plane : voxel_map->planes)
        {
            *prepare_key_cloud += *projectCornerNMS(plane);
        }
        
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr candidates = nms3D(prepare_key_cloud);

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
        Eigen::Matrix3f rot;
        Eigen::Vector3f trans;
        rot.row(0) = plane.norms.col(2).cast<float>();
        rot.row(1) = plane.norms.col(1).cast<float>();
        rot.row(2) = plane.norms.col(0).cast<float>();
        trans = -rot * plane.mean.cast<float>();
        pcl::transformPointCloud(*plane.corner_cloud, *filtered_point, trans, Eigen::Quaternionf(rot));

        std::unordered_map<VoxelKey, pcl::PointXYZINormal, VoxelKey::Hasher> image_grids;
        // 投影到平面
        for (int i = 0; i < filtered_point->size(); i++)
        {
            pcl::PointXYZINormal &p = filtered_point->points[i];

            Eigen::Vector3d point_vec(p.x, p.y, p.z);

            point_vec = (point_vec / image_resolution).array().floor();

            VoxelKey k(static_cast<int64_t>(point_vec(0)), static_cast<int64_t>(point_vec(1)), 0);

            if (image_grids.find(k) == image_grids.end())
            {
                image_grids[k] = plane.corner_cloud->points[i];
                image_grids[k].curvature = 1.0;
                image_grids[k].normal_x = p.x;
                image_grids[k].normal_y = p.y;
                image_grids[k].normal_z = p.z;
            }
            else
            {
                if (std::abs(p.z) > std::abs(image_grids[k].normal_z))
                {
                    image_grids[k] = plane.corner_cloud->points[i];
                    image_grids[k].curvature = 1.0;
                    image_grids[k].normal_x = p.x;
                    image_grids[k].normal_y = p.y;
                    image_grids[k].normal_z = p.z;
                }
            }
        }
        // 在一定范围内进行非极大值抑制
        for (auto it = image_grids.begin(); it != image_grids.end(); it++)
        {
            VoxelKey ck = it->first;
            pcl::PointXYZINormal &p = it->second;
            if (p.curvature == 0.0)
                continue;
            for (VoxelKey &nk : nears2d(ck, nms_range))
            {
                if (image_grids.find(nk) == image_grids.end())
                    continue;
                if (std::abs(p.normal_z) > std::abs(image_grids[nk].normal_z))
                {
                    image_grids[nk].curvature == 0.0;
                }
                else if (std::abs(image_grids[nk].normal_z) > std::abs(p.normal_z))
                {
                    p.curvature = 0.0;
                }
            }
        }

        filtered_point->clear();

        // 修正剩余点的法向量
        for (auto it = image_grids.begin(); it != image_grids.end(); it++)
        {
            if (it->second.curvature == 1.0)
            {
                it->second.intensity = std::abs(it->second.normal_z);
                it->second.normal_x = plane.norms.col(0).x();
                it->second.normal_y = plane.norms.col(0).y();
                it->second.normal_z = plane.norms.col(0).z();
                filtered_point->push_back(it->second);
            }
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

} // namespace stdes
