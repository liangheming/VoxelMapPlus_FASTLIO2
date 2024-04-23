#include "voxel_map.h"

namespace lio
{

    uint64_t VoxelGrid::count = 0;

    VoxelGrid::VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map)
        : max_point_thresh(_max_point_thresh),
          update_point_thresh(_update_point_thresh),
          plane_thresh(_plane_thresh),
          position(_position.x, position.y, position.z),
          map(_map)
    {
        group_id = VoxelGrid::count++;
        is_init = false;
        is_plane = false;
        temp_points.reserve(max_point_thresh);
        newly_add_point = 0;
        plane = std::make_shared<Plane>();
        update_enable = true;
    }

    void VoxelGrid::addToPlane(const PointWithCov &pv)
    {
        plane->mean = plane->mean + (pv.point - plane->mean) / (plane->n + 1.0);
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
                    if (temp_points.size() > max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    // do merge
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
                    if (temp_points.size() > max_point_thresh)
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

            Eigen::Matrix<double, 4, 3> J;
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            for (int m = 1; m < 3; m++)
            {
                Eigen::Matrix<double, 1, 3> F_m = (pv.point - plane->mean).transpose() / ((plane->n) * (evals(0) - evals(m))) *
                                                  (evecs.col(m) * plane_norm.transpose() +
                                                   plane_norm * evecs.col(m).transpose());
                F.row(m) = F_m;
            }
            J.block<3, 3>(0, 0) = evecs * F;
            // J.block<1, 3>(3, 0) = plane_norm.transpose() * J_Q + plane->mean.transpose() * J.block<3, 3>(0, 0);
            J.block<1, 3>(3, 0) = plane_norm.transpose() * J_Q;
            plane->plane_cov += J * pv.cov * J.transpose();
        }
        double axis_distance = -plane->mean.dot(plane_norm);
        if (axis_distance < 0)
        {
            plane_norm = -plane_norm;
            axis_distance = -axis_distance;
        }
        plane->plane_param.head(3) = plane_norm;
        plane->plane_param(3) = axis_distance;
    }

    VoxelMap::VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size) : max_point_thresh(_max_point_thresh), update_point_thresh(_update_point_thresh), plane_thresh(_plane_thresh), voxel_size(_voxel_size)
    {
        featmap.clear();
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
            }
            featmap[k]->pushPoint(pv);
        }
    }

    void VoxelMap::buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid)
    {
        data.is_valid = false;
        if (voxel_grid->is_plane)
        {
            Eigen::Vector4d homo_point(data.point_world.x(), data.point_world.y(), data.point_world.z(), 1.0);
            data.plane_param = voxel_grid->plane->plane_param;
            data.plane_cov = voxel_grid->plane->plane_cov;
            data.residual = homo_point.dot(data.plane_param);
            double sigma_l = homo_point.transpose() * data.plane_cov * homo_point;
            Eigen::Vector3d plane_norm(data.plane_param(0), data.plane_param(1), data.plane_param(2));
            sigma_l += plane_norm.transpose() * data.cov_world * plane_norm;
            if (std::abs(data.residual) > 3.0 * sqrt(sigma_l))
            {
                data.is_valid = false;
            }
            else
            {
                data.is_valid = true;
            }
        }
    }

} // namespace lio
