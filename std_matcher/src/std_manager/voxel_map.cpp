#include "voxel_map.h"

namespace stdes
{
    VoxelGrid::VoxelGrid(double _plane_thesh, int _min_num_thresh) : plane_thresh(_plane_thesh), min_num_thresh(_min_num_thresh)
    {
        mean.setZero();
        ppt.setZero();
        clouds.clear();
        is_plane = false;
    }

    void VoxelGrid::addPoint(const pcl::PointXYZINormal &p)
    {
        Eigen::Vector3d point(p.x, p.y, p.z);
        mean += (point - mean) / (clouds.size() + 1.0);
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

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr VoxelMap::coloredPlaneCloud()
    {
        int size = planes.size();
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_visual(new pcl::PointCloud<pcl::PointXYZRGBA>);
        for (int i = 0; i < size; i++)
        {
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
        return cloud_visual;
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

        // std::cout << "plane voxel size: " << plane_voxels.size() << " total voxel size: " << voxels.size() << std::endl;
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
            std::queue<VoxelKey> buffer;
            buffer.push(k);
            double min_lambda = voxels[k]->lamdas(0);

            while (!buffer.empty())
            {

                VoxelKey ck = buffer.front();
                buffer.pop();
                std::shared_ptr<VoxelGrid> cv = voxels[ck];
                flags[ck] = false;
                *p.sur_cloud += cv->clouds;
                if (cv->lamdas(0) <= min_lambda)
                {
                    p.lamdas = cv->lamdas;
                    p.norms = cv->norms;
                    min_lambda = cv->lamdas[0];
                }
                p.num += 1;

                for (VoxelKey &nk : sixNears(ck))
                {
                    auto n_it = voxels.find(nk);
                    if (n_it == voxels.end())
                        continue;
                    if (n_it->second->is_plane)
                    {

                        // 如果是平面，且可以合并
                        if (flags[n_it->first] && ((cv->norms.col(0) - n_it->second->norms.col(0)).norm() < merge_thresh || (cv->norms.col(0) + n_it->second->norms.col(0)).norm() < merge_thresh))
                        {
                            buffer.push(n_it->first);
                        }
                    }
                    else
                    {
                        // 如果不是平面则添加到cornner里面
                        if (n_it->second->clouds.size() > min_num_thresh)
                            *p.corner_cloud += n_it->second->clouds;
                    }
                }
            }

            if (p.num > 1 && p.corner_cloud->size() > min_num_thresh)
                planes.push_back(p);
        }
    }
} // namespace stdes
