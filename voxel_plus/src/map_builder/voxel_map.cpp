#include "voxel_map.h"
#include <iostream>
namespace lio
{
    OctoTree::OctoTree(int _max_layer, int _layer, std::vector<int> _update_size_threshes, int _max_point_thresh, double _plane_thresh)
        : max_layer(_max_layer), layer(_layer), update_size_threshes(_update_size_threshes), max_point_thresh(_max_point_thresh), plane_thresh(_plane_thresh)
    {
        temp_points.clear();
        is_leave = false;
        all_point_num = 0;
        new_point_num = 0;
        update_size_thresh_for_new = 5;
        is_initialized = false;
        update_enable = true;
        update_size_thresh = update_size_threshes[layer];
        plane.is_valid = false;
        leaves.resize(8, nullptr);
    }
    
    void OctoTree::insert(const std::vector<PointWithCov> &input_points)
    {
        if (!is_initialized)
        {
            temp_points.insert(temp_points.begin(), input_points.begin(), input_points.end());
            all_point_num += input_points.size();
            new_point_num += input_points.size();
            initialize_tree();
            return;
        }

        if (is_leave)
        {
            if (update_enable)
            {
                temp_points.insert(temp_points.begin(), input_points.begin(), input_points.end());
                all_point_num += input_points.size();
                new_point_num += input_points.size();
                if (new_point_num >= update_size_thresh_for_new)
                {
                    build_plane(temp_points);
                    new_point_num = 0;
                }

                if (all_point_num >= max_point_thresh)
                {
                    update_enable = false;
                    std::vector<PointWithCov>().swap(temp_points);
                }
            }
            return;
        }

        if (layer < max_layer - 1)
        {
            if (temp_points.size() != 0)
                std::vector<PointWithCov>().swap(temp_points);
            std::vector<std::vector<PointWithCov>> package(8, std::vector<PointWithCov>(0));

            for (size_t i = 0; i < input_points.size(); i++)
            {
                int xyz[3] = {0, 0, 0};
                int leafnum = subIndex(input_points[i], xyz);
                if (leaves[leafnum] == nullptr)
                {

                    leaves[leafnum] = std::make_shared<OctoTree>(max_layer, layer + 1, update_size_threshes, max_point_thresh, plane_thresh);
                    Eigen::Vector3d shift((2 * xyz[0] - 1) * quater_length, (2 * xyz[1] - 1) * quater_length, (2 * xyz[2] - 1) * quater_length);
                    leaves[leafnum]->center = center + shift;
                    leaves[leafnum]->quater_length = quater_length / 2;
                }
                package[leafnum].push_back(input_points[i]);
            }

            for (int i = 0; i < 8; i++)
            {
                if (package[i].size() == 0)
                    continue;
                leaves[i]->insert(package[i]);
            }
        }
    }

    void OctoTree::initialize_tree()
    {
        if (all_point_num < update_size_thresh)
            return;
        is_initialized = true;
        new_point_num = 0;
        build_plane(temp_points);

        if (plane.is_valid)
        {
            is_leave = true;
            if (temp_points.size() > max_point_thresh)
            {
                update_enable = false;
                std::vector<PointWithCov>().swap(temp_points);
            }
            else
            {
                update_enable = true;
            }
        }
        else
        {
            split_tree();
        }
    }

    void OctoTree::split_tree()
    {
        if (layer >= max_layer - 1)
        {
            is_leave = true;
            return;
        }

        std::vector<std::vector<PointWithCov>> package(8, std::vector<PointWithCov>(0));

        for (size_t i = 0; i < temp_points.size(); i++)
        {
            int xyz[3] = {0, 0, 0};
            int leafnum = subIndex(temp_points[i], xyz);
            if (leaves[leafnum] == nullptr)
            {

                leaves[leafnum] = std::make_shared<OctoTree>(max_layer, layer + 1, update_size_threshes, max_point_thresh, plane_thresh);
                Eigen::Vector3d shift((2 * xyz[0] - 1) * quater_length, (2 * xyz[1] - 1) * quater_length, (2 * xyz[2] - 1) * quater_length);
                leaves[leafnum]->center = center + shift;
                leaves[leafnum]->quater_length = quater_length / 2;
            }
            package[leafnum].push_back(temp_points[i]);
        }

        for (int i = 0; i < 8; i++)
        {
            if (package[i].size() == 0)
                continue;
            leaves[i]->insert(package[i]);
        }
        std::vector<PointWithCov>().swap(temp_points);
    }

    int OctoTree::subIndex(const PointWithCov &pv, int *xyz)
    {
        if (pv.point[0] > center[0])
            xyz[0] = 1;
        if (pv.point[1] > center[1])
            xyz[1] = 1;
        if (pv.point[2] > center[2])
            xyz[2] = 1;
        return 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    }

    void OctoTree::build_plane(const std::vector<PointWithCov> &points)
    {
        plane.plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
        plane.covariance = Eigen::Matrix3d::Zero();
        plane.center = Eigen::Vector3d::Zero();
        plane.normal = Eigen::Vector3d::Zero();
        plane.points_size = points.size();

        for (auto pv : points)
        {
            plane.covariance += pv.point * pv.point.transpose();
            plane.center += pv.point;
        }
        plane.center = plane.center / plane.points_size;
        plane.covariance = plane.covariance / plane.points_size - plane.center * plane.center.transpose();

        Eigen::EigenSolver<Eigen::Matrix3d> es(plane.covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal = evals.real();
        Eigen::Matrix3d::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;

        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane.points_size);
        plane.eigens << evalsReal(evalsMin), evalsReal(evalsMid), evalsReal(evalsMax);
        plane.normal = evecs.real().col(evalsMin);
        plane.x_normal = evecs.real().col(evalsMax);
        plane.y_normal = evecs.real().col(evalsMid);

        if (plane.eigens[0] < plane_thresh)
        {
            for (int i = 0; i < points.size(); i++)
            {
                Eigen::Matrix<double, 6, 3> J;
                Eigen::Matrix3d F;
                for (int m = 0; m < 3; m++)
                {
                    if (m != (int)evalsMin)
                    {
                        Eigen::Matrix<double, 1, 3> F_m =
                            (points[i].point - plane.center).transpose() /
                            ((plane.points_size) * (evalsReal[evalsMin] - evalsReal[m])) *
                            (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                             evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                        F.row(m) = F_m;
                    }
                    else
                    {
                        Eigen::Matrix<double, 1, 3> F_m;
                        F_m << 0, 0, 0;
                        F.row(m) = F_m;
                    }
                }
                J.block<3, 3>(0, 0) = evecs.real() * F;
                J.block<3, 3>(3, 0) = J_Q;
                plane.plane_cov += J * points[i].cov * J.transpose();
            }
            plane.is_valid = true;
        }
        else
        {
            plane.is_valid = false;
        }
    }

    void VoxelMap::pack(const std::vector<PointWithCov> &input_points)
    {
        sub_map.clear();
        uint plsize = input_points.size();
        for (uint i = 0; i < plsize; i++)
        {
            const PointWithCov &p_v = input_points[i];
            VoxelKey k = index(p_v.point);
            auto it = feat_map.find(k);
            auto sub_it = sub_map.find(k);
            if (it == feat_map.end())
            {
                if (sub_it == sub_map.end())
                {
                    // sub_map[k] = VoxelGrid();
                    sub_map[k].type = SubVoxelType::INSERT;
                }
                sub_map[k].points.push_back(p_v);
            }
            else
            {
                if (!feat_map[k].tree->update_enable)
                    continue;
                if (sub_it == sub_map.end())
                {
                    // sub_map[k] = VoxelGrid();
                    sub_map[k].type = SubVoxelType::UPDATE;
                    sub_map[k].it = feat_map[k].it;
                }
                sub_map[k].points.push_back(p_v);
            }
        }
    }

    void VoxelMap::insert(const std::vector<PointWithCov> &input_points)
    {
        pack(input_points);

        for (auto &pair : sub_map)
        {
            if (pair.second.type == SubVoxelType::INSERT)
            {
                cache.push_front(pair.first);
                pair.second.it = cache.begin();
                // feat_map[pair.first] = VoxelValue();
                feat_map[pair.first].tree = std::make_shared<OctoTree>(max_layer, 0, update_size_threshes, max_point_thresh, plane_thresh);
                feat_map[pair.first].it = pair.second.it;
                feat_map[pair.first].tree->center = Eigen::Vector3d((0.5 + pair.first.x) * voxel_size, (0.5 + pair.first.y) * voxel_size, (0.5 + pair.first.z) * voxel_size);
                feat_map[pair.first].tree->quater_length = voxel_size / 4;
                feat_map[pair.first].tree->insert(pair.second.points);
                if (cache.size() > capacity)
                {
                    feat_map.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                cache.splice(cache.begin(), cache, pair.second.it);
                feat_map[pair.first].tree->insert(pair.second.points);
            }
        }
    }

    VoxelKey VoxelMap::index(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    void VoxelMap::buildResidual(ResidualData &info, std::shared_ptr<OctoTree> oct_tree)
    {
        if (oct_tree->plane.is_valid)
        {
            Eigen::Vector3d p_world_to_center = info.point_world - oct_tree->plane.center;
            info.plane_center = oct_tree->plane.center;
            info.plane_norm = oct_tree->plane.normal;
            info.plane_cov = oct_tree->plane.plane_cov;
            info.residual = info.plane_norm.transpose() * p_world_to_center;
            double dis_to_plane = std::abs(info.residual);
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_world_to_center;
            J_nq.block<1, 3>(0, 3) = -info.plane_norm;
            double sigma_l = J_nq * info.plane_cov * J_nq.transpose();
            sigma_l += info.plane_norm.transpose() * info.cov * info.plane_norm;
            if (dis_to_plane < info.sigma_num * sqrt(sigma_l))
            {
                info.is_valid = true;
            }
        }
        else
        {
            if (info.current_layer < max_layer - 1)
            {
                for (size_t i = 0; i < 8; i++)
                {
                    if (oct_tree->leaves[i] == nullptr)
                        continue;
                    info.current_layer += 1;
                    buildResidual(info, oct_tree->leaves[i]);
                    if (info.is_valid)
                        break;
                }
            }
        }
    }
} // namespace lio
