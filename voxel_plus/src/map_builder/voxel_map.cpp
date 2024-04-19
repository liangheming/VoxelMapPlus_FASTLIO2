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

        // Eigen::Matrix4d mat;
        // mat << plane->xx, plane->xy, plane->xz, plane->x,
        //     plane->xy, plane->yy, plane->yz, plane->y,
        //     plane->xz, plane->yz, plane->zz, plane->z,
        //     plane->x, plane->y, plane->z, static_cast<double>(plane->n);

        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> pes(mat);
        // Eigen::Matrix4d eigen_vec = pes.eigenvectors();
        // Eigen::Vector4d eigen_val = pes.eigenvalues();

        // double min_eigen = eigen_val(0);

        // if (min_eigen > plane_thesh)
        // {
        //     plane->is_plane = false;
        //     return;
        // }

        // plane->is_plane = true;
        // Eigen::Vector4d p_param = eigen_vec.col(0);
        // double a = p_param(0), b = p_param(1), c = p_param(2), d = p_param(3);
        // double p_norm = Eigen::Vector3d(p_param(0), p_param(1), p_param(2)).norm();

        // double sq_p_norm = p_norm * p_norm;
        // plane->plane_param = p_param / p_norm;

        // if (plane->plane_param(3) < 0)
        //     plane->plane_param = -plane->plane_param;
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
