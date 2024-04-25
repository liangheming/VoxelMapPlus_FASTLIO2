#include "voxel_map.h"

namespace lio
{

    uint64_t VoxelGrid::count = 0;

    double VoxelGrid::merge_thresh_for_angle = 0.1;
    double VoxelGrid::merge_thresh_for_distance = 0.04;

    VoxelGrid::VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map)
        : max_point_thresh(_max_point_thresh),
          update_point_thresh(_update_point_thresh),
          plane_thresh(_plane_thresh),
          position(_position.x, _position.y, _position.z)
    {
        merged = false;
        group_id = VoxelGrid::count++;
        is_init = false;
        is_plane = false;
        temp_points.reserve(max_point_thresh);
        newly_add_point = 0;
        plane = std::make_shared<Plane>();
        update_enable = true;
        map = _map;
        center = Eigen::Vector3d(position.x + 0.5, position.y + 0.5, position.z + 0.5) * map->voxel_size;
    }

    void VoxelGrid::addToPlane(const PointWithCov &pv)
    {
        plane->mean += (pv.point - plane->mean) / (plane->n + 1.0);
        plane->ppt += pv.point * pv.point.transpose();
        plane->n += 1;
    }

    void VoxelGrid::addPoint(const PointWithCov &pv)
    {
        addToPlane(pv);
        temp_points.push_back(pv);
    }

    void VoxelGrid::pushPoint(const PointWithCov &pv)
    {
        if (!is_init)
        {
            addToPlane(pv);
            temp_points.push_back(pv);
            updatePlane();
        }
        else
        {
            if (is_plane)
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    if (temp_points.size() >= max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    merge();
                }
            }
            else
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    if (temp_points.size() >= max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
            }
        }
    }

    void VoxelGrid::updatePlane()
    {
        assert(temp_points.size() == plane->n);
        if (plane->n < update_point_thresh)
            return;
        is_init = true;
        Eigen::Matrix3d cov = plane->ppt / static_cast<double>(plane->n) - plane->mean * plane->mean.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix3d evecs = es.eigenvectors();
        Eigen::Vector3d evals = es.eigenvalues();
        if (evals(0) > plane_thresh)
        {
            is_plane = false;
            return;
        }
        is_plane = true;
        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane->n);
        Eigen::Vector3d plane_norm = evecs.col(0);
        for (PointWithCov &pv : temp_points)
        {

            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            for (int m = 1; m < 3; m++)
            {
                Eigen::Matrix<double, 1, 3> F_m = (pv.point - plane->mean).transpose() / ((plane->n) * (evals(0) - evals(m))) *
                                                  (evecs.col(m) * plane_norm.transpose() +
                                                   plane_norm * evecs.col(m).transpose());
                F.row(m) = F_m;
            }
            J.block<3, 3>(0, 0) = evecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->cov += J * pv.cov * J.transpose();
        }
        double axis_distance = -plane->mean.dot(plane_norm);
        if (axis_distance < 0.0)
            plane_norm = -plane_norm;
        plane->norm = plane_norm;
        // center = plane->mean;
    }

    void VoxelGrid::merge()
    {
        std::vector<VoxelKey> near;
        near.push_back(VoxelKey(position.x - 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y - 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z - 1));
        near.push_back(VoxelKey(position.x + 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y + 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z + 1));

        for (VoxelKey &k : near)
        {
            auto it = map->featmap.find(k);
            if (it != map->featmap.end())
            {
                std::shared_ptr<VoxelGrid> near_node = it->second;
                if (near_node->group_id == group_id || near_node->update_enable || !near_node->is_plane)
                    continue;
                double norm_distance = 1.0 - near_node->plane->norm.dot(plane->norm);
                double axis_distance = std::abs(near_node->plane->norm.dot(near_node->plane->mean) - plane->norm.dot(plane->mean));

                if (norm_distance > merge_thresh_for_angle || axis_distance > merge_thresh_for_distance)
                    continue;
                double tn0 = plane->cov.block<3, 3>(0, 0).trace(),
                       tm0 = plane->cov.block<3, 3>(3, 3).trace(),
                       tn1 = near_node->plane->cov.block<3, 3>(0, 0).trace(),
                       tm1 = near_node->plane->cov.block<3, 3>(3, 3).trace();
                double tc0 = tn0 + tm0, tc1 = tn1 + tm1;
                Eigen::Vector3d new_mean = tm0 * near_node->plane->mean + tm1 * plane->mean / (tm0 + tm1);
                Eigen::Vector3d new_norm = tn0 * near_node->plane->norm + tn1 * plane->norm / (tn0 + tn1);
                Eigen::Matrix<double, 6, 6> new_cov = (tc0 * tc0 * near_node->plane->cov + tc1 * tc1 * plane->cov) / ((tc0 + tc1) * (tc0 + tc1));

                near_node->group_id = group_id;
                merged = true;
                near_node->merged = true;

                if (-new_mean.dot(new_norm) < 0.0)
                    new_norm = -new_norm;

                plane->mean = new_mean;
                plane->norm = new_norm;
                plane->cov = new_cov;

                near_node->plane->mean = new_mean;
                near_node->plane->norm = new_norm;
                near_node->plane->cov = new_cov;
            }
        }
    }

    VoxelMap::VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size, int _capacity) : max_point_thresh(_max_point_thresh), update_point_thresh(_update_point_thresh), plane_thresh(_plane_thresh), voxel_size(_voxel_size), capacity(_capacity)
    {
        featmap.clear();
        cache.clear();
    }

    VoxelKey VoxelMap::index(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    void VoxelMap::build(std::vector<PointWithCov> &pvs)
    {
        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();

                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }

            featmap[k]->addPoint(pv);
        }

        for (auto it = featmap.begin(); it != featmap.end(); it++)
        {
            it->second->updatePlane();
        }
    }

    void VoxelMap::update(std::vector<PointWithCov> &pvs)
    {

        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();
                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }
            featmap[k]->pushPoint(pv);
        }
    }

    bool VoxelMap::buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid)
    {
        data.is_valid = false;
        if (voxel_grid->is_plane)
        {
            Eigen::Vector3d p2m = (data.point_world - voxel_grid->plane->mean);
            data.plane_norm = voxel_grid->plane->norm;
            data.plane_mean = voxel_grid->plane->mean;
            data.residual = data.plane_norm.dot(p2m);
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p2m;
            J_nq.block<1, 3>(0, 3) = -data.plane_norm;
            double sigma_l = J_nq * data.plane_cov * J_nq.transpose();
            sigma_l += data.plane_norm.transpose() * data.cov_world * data.plane_norm;
            if (std::abs(data.residual) < 3.0 * sqrt(sigma_l))
                data.is_valid = true;
        }
        return data.is_valid;
    }

} // namespace lio
