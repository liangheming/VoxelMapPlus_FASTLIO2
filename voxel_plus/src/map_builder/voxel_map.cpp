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
        // total_point_num = 0;
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

    void UnionFindNode::addToPlane(const PointWithCov &pv)
    {

        plane->mean = plane->mean + (pv.point - plane->mean) / (plane->n + 1.0);
        plane->ppt += pv.point * pv.point.transpose();
        double x = pv.point.x(), y = pv.point.y(), z = pv.point.z();

        plane->x += x;
        plane->y += y;
        plane->z += z;

        plane->xx += (x * x);
        plane->yy += (y * y);
        plane->zz += (z * z);

        plane->xy += (x * y);
        plane->xz += (x * z);
        plane->yz += (y * z);

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
        Eigen::Vector3d evals = es.eigenvalues();
        double mean_eigen = evals.minCoeff();
        if (mean_eigen > plane_thesh)
            return;

        plane->is_plane = true;
        Eigen::Matrix4d mat;
        mat << plane->xx, plane->xy, plane->xz, plane->x,
            plane->xy, plane->yy, plane->yz, plane->y,
            plane->xz, plane->yz, plane->zz, plane->z,
            plane->x, plane->y, plane->z, static_cast<double>(plane->n);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> pes(mat);
        Eigen::Matrix4d eigen_vec = pes.eigenvectors();
        Eigen::Vector4d eigen_val = pes.eigenvalues();
        Eigen::Vector4d::Index min_idx;
        eigen_val.minCoeff(&min_idx);
        Eigen::Vector4d p_param = eigen_vec.col(min_idx);
        double a = p_param(0), b = p_param(1), c = p_param(2), d = p_param(3);

        double p_norm = Eigen::Vector3d(p_param(0), p_param(1), p_param(2)).norm();
        double sq_p_norm = p_norm * p_norm;
        plane->plane_param = p_param / p_norm;
        Eigen::Matrix4d mat_inv = eigen_vec * Eigen::Vector4d((eigen_val.array() > 1e-6).select(eigen_val.array().inverse(), 0)).asDiagonal() * eigen_vec.transpose();

        Eigen::Matrix4d derive_param;

        derive_param << p_norm - a * a / p_norm, -a * b / p_norm, -a * c / p_norm, 0.0,
            -a * b / p_norm, p_norm - b * b / p_norm, -a * c / p_norm, 0.0,
            -a * c / p_norm, -b * c / p_norm, p_norm - c * c / p_norm, 0.0,
            -a * d / p_norm, -b * d / p_norm, -c * d / p_norm, p_norm;
        derive_param /= sq_p_norm;

        plane->plane_cov.setZero();

        for (PointWithCov &pv : temp_points)
        {
            double x = pv.point(0), y = pv.point(1), z = pv.point(2);
            Eigen::Matrix<double, 4, 3> dudp = Eigen::Matrix<double, 4, 3>::Zero();
            for (int i = 0; i < 4; i++)
            {
                if (static_cast<int>(min_idx) == i)
                    continue;
                Eigen::Matrix4d dmatdx;
                dmatdx << 2 * x, y, z, 1.0,
                    y, 0.0, 0.0, 0.0,
                    z, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0;
                dudp.col(0) += (eigen_vec.col(i) * eigen_vec.col(i).transpose() * dmatdx * eigen_vec.col(min_idx)) / (eigen_val(min_idx) - eigen_val(i));
                
                dmatdx.setZero();
                dmatdx << 0.0, x, 0.0, 0.0,
                    x, 2 * y, z, 1.0,
                    0.0, z, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0;
                dudp.col(1) += (eigen_vec.col(i) * eigen_vec.col(i).transpose() * dmatdx * eigen_vec.col(min_idx)) / (eigen_val(min_idx) - eigen_val(i));

                dmatdx.setZero();
                dmatdx << 0.0, 0.0, x, 0.0,
                    0.0, 0.0, y, 0.0,
                    x, y, 2 * z, 1.0,
                    0.0, 0.0, 1.0, 0.0;
                dudp.col(2) += (eigen_vec.col(i) * eigen_vec.col(i).transpose() * dmatdx * eigen_vec.col(min_idx)) / (eigen_val(min_idx) - eigen_val(i));
            }
            plane->plane_cov += derive_param * dudp * pv.cov * dudp.transpose() * derive_param.transpose();
            
            // make d>0.0;
            if (plane->plane_param(3) < 0)
                plane->plane_param = -plane->plane_param;
        }

        // std::cout << sq_p_norm << std::endl;
        // std::cout << "get_derive_param: " << temp_points.size() << std::endl;
        // std::cout << (p_param / p_norm).transpose() << std::endl;
        // std::cout << plane->plane_cov << std::endl;
        // std::cout << "=========================================" << std::endl;
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

    VoxelMap::~VoxelMap()
    {
        if (feat_map.size() == 0)
            return;
        for (auto it = feat_map.begin(); it != feat_map.end(); it++)
        {
            delete it->second;
        }
        FeatMap temp;
        feat_map.swap(temp);
    }
} // namespace lio
