#include "voxel_map.h"

namespace lio
{

    UnionFindNode::UnionFindNode(double _plane_thresh, int _update_size_thresh, int _max_point_thresh, Eigen::Vector3d &_voxel_center, VoxelMap *_map)
        : plane_thesh(_plane_thresh),
          update_size_thresh(_update_size_thresh),
          max_point_thresh(_max_point_thresh),
          voxel_center(_voxel_center),
          map(_map)
    {
        newly_added_num = 0;
        update_enable = true;
        plane = std::make_shared<Plane>();
        root_node = this;
        temp_points.clear();
    }

    void UnionFindNode::push(const PointWithCov &pv)
    {
        addToPlane(pv);
        temp_points.push_back(pv);
    }

    void UnionFindNode::emplace(const PointWithCov &pv)
    {
        if (!isInitialized())
        {
            addToPlane(pv);
            temp_points.push_back(pv);
            updatePlane();
        }
        else
        {
            if (isPlane())
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_added_num++;
                    if (newly_added_num >= update_size_thresh)
                    {
                        updatePlane();
                        newly_added_num = 0;
                    }
                    if (temp_points.size() > max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    // try merge
                    // std::cout << "is plane  and  get ready to be merge" << std::endl;
                }
            }
            else
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_added_num++;
                    if (newly_added_num >= update_size_thresh)
                    {
                        updatePlane();
                        newly_added_num = 0;
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

    void UnionFindNode::addToPlane(const PointWithCov &pv)
    {

        plane->mean = plane->mean + (pv.point - plane->mean) / (plane->n + 1.0);
        plane->ppt += pv.point * pv.point.transpose();
        // double x = pv.point.x(), y = pv.point.y(), z = pv.point.z();

        // plane->x += x;
        // plane->y += y;
        // plane->z += z;

        // plane->xx += (x * x);
        // plane->yy += (y * y);
        // plane->zz += (z * z);

        // plane->xy += (x * y);
        // plane->xz += (x * z);
        // plane->yz += (y * z);

        plane->n += 1;
    }

    void UnionFindNode::updatePlane()
    {
        assert(temp_points.size() == plane->n);
        if (plane->n < update_size_thresh)
            return;

        plane->is_init = true;

        Eigen::Matrix3d cov = plane->ppt / static_cast<double>(plane->n) - plane->mean * plane->mean.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);

        Eigen::Vector3d eigen_vals = es.eigenvalues();
        Eigen::Matrix3d eigen_vecs = es.eigenvectors();

        if (eigen_vals(0) > plane_thesh)
        {
            plane->is_plane = false;
            return;
        }

        plane->is_plane = true;
        plane->norm_vec = eigen_vecs.col(0);

        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane->n);

        for (int i = 0; i < temp_points.size(); i++)
        {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++)
            {
                if (m == 0)
                {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
                else
                {
                    Eigen::Matrix<double, 1, 3> F_m =
                        (temp_points[i].point - plane->mean).transpose() /
                        ((plane->n) * (eigen_vals(0) - eigen_vals(m))) *
                        (eigen_vecs.col(m) * eigen_vecs.col(0).transpose() +
                         eigen_vecs.col(0) * eigen_vecs.col(m).transpose());
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = eigen_vecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->plane_cov += J * temp_points[i].cov * J.transpose();
        }
    }

    VoxelMap::VoxelMap(double _voxel_size, double _plane_thresh, int _update_size_thresh, int _max_point_thresh)
        : voxel_size(_voxel_size),
          plane_thresh(_plane_thresh),
          update_size_thresh(_update_size_thresh),
          max_point_thresh(_max_point_thresh)
    {
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
            auto f_it = feat_map.find(k);
            if (f_it == feat_map.end())
            {
                Eigen::Vector3d center(static_cast<double>(k.x) + 0.5, static_cast<double>(k.y) + 0.5, static_cast<double>(k.z) + 0.5);
                center *= voxel_size;
                feat_map[k] = new UnionFindNode(plane_thresh, update_size_thresh, max_point_thresh, center, this);
            }
            feat_map[k]->push(pv);
        }
        for (auto it = feat_map.begin(); it != feat_map.end(); it++)
        {
            it->second->updatePlane();
        }
    }

    void VoxelMap::update(std::vector<PointWithCov> &pvs)
    {
        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto f_it = feat_map.find(k);
            if (f_it == feat_map.end())
            {
                Eigen::Vector3d center(static_cast<double>(k.x) + 0.5, static_cast<double>(k.y) + 0.5, static_cast<double>(k.z) + 0.5);
                center *= voxel_size;
                feat_map[k] = new UnionFindNode(plane_thresh, update_size_thresh, max_point_thresh, center, this);
            }
            feat_map[k]->emplace(pv);
        }
    }

    void VoxelMap::buildResidual(ResidualData &info, UnionFindNode *node)
    {
        info.is_valid = false;
        if (node->isPlane())
        {
            info.is_valid = true;
            info.residual = node->plane->norm_vec.dot(info.point_world - node->plane->mean);
            info.plane_mean = node->plane->mean;
            info.plane_norm = node->plane->norm_vec;
            info.plane_cov = node->plane->plane_cov;

            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = info.point_world - node->plane->mean;
            J_nq.block<1, 3>(0, 3) = -info.plane_norm;
            double sigma_l = J_nq * node->plane->plane_cov * J_nq.transpose();
            sigma_l += info.plane_norm.transpose() * info.cov_world * info.plane_norm;

            // if (std::abs(info.residual) > 3 * sqrt(sigma_l))
            // {
            //     info.is_valid = false;
            // }
        }
    }

    VoxelMap::~VoxelMap()
    {
        if (feat_map.size() == 0)
            return;
        for (auto it = feat_map.begin(); it != feat_map.end(); it++)
        {
            delete it->second;
        }
        FeatMap().swap(feat_map);
    }
} // namespace lio
