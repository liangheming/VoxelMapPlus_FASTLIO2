#include "utils.h"

void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double range_min, double range_max)
{
    int point_num = msg->point_num;
    out->clear();
    out->reserve(point_num / filter_num + 1);
    uint valid_num = 0;
    for (uint i = 0; i < point_num; i++)
    {
        if ((msg->points[i].line < 4) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            if ((valid_num++) % filter_num != 0)
                continue;
            pcl::PointXYZINormal p;
            p.x = msg->points[i].x;
            p.y = msg->points[i].y;
            p.z = msg->points[i].z;
            p.intensity = msg->points[i].reflectivity;
            p.curvature = msg->points[i].offset_time / float(1000000); // 纳秒->毫秒
            double sq_range = p.x * p.x + p.y * p.y + p.z * p.z;
            if (sq_range > (range_min * range_min) && sq_range < range_max * range_max)
            {
                out->push_back(p);
            }
        }
    }
}

sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp)
{
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*inp, msg);
    if (timestamp < 0)
        msg.header.stamp = ros::Time().now();
    else
        msg.header.stamp = ros::Time().fromSec(timestamp);
    msg.header.frame_id = frame_id;
    return msg;
}

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);

    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.header.stamp = ros::Time().fromSec(timestamp);
    odom.child_frame_id = child_frame_id;
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    odom.pose.pose.position.x = pos(0);
    odom.pose.pose.position.y = pos(1);
    odom.pose.pose.position.z = pos(2);

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    return odom;
}

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin)
    {
        v = vmin;
    }

    if (v > vmax)
    {
        v = vmax;
    }

    double dr, dg, db;

    if (v < 0.1242)
    {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    }
    else if (v < 0.3747)
    {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    }
    else if (v < 0.6253)
    {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    }
    else if (v < 0.8758)
    {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    }
    else
    {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t)(255 * dr);
    g = (uint8_t)(255 * dg);
    b = (uint8_t)(255 * db);
}

void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q)
{
    Eigen::Matrix3d rot;
    rot.col(0) = x_vec;
    rot.col(1) = y_vec;
    rot.col(2) = z_vec;
    Eigen::Quaterniond eq(rot);
    eq.normalize();
    q.w = eq.w();
    q.x = eq.x();
    q.y = eq.y();
    q.z = eq.z();
}

void calcVectQuation(const Eigen::Vector3d &norm_vec, geometry_msgs::Quaternion &q)
{
    Eigen::Quaterniond rq = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1), norm_vec);
    q.w = rq.w();
    q.x = rq.x();
    q.y = rq.y();
    q.z = rq.z();
}

visualization_msgs::MarkerArray voxel2MarkerArray(std::shared_ptr<lio::VoxelMap> map, const std::string &frame_id, const double &timestamp, int max_capacity, double voxel_size)
{
    visualization_msgs::MarkerArray voxel_plane;
    int size = std::min(static_cast<int>(map->cache.size()), max_capacity);

    voxel_plane.markers.reserve(size);
    int count = 0;
    for (auto &k : map->cache)
    {
        if (count >= size)
            break;
        std::shared_ptr<lio::VoxelGrid> grid = map->featmap[k];
        if (!grid->is_plane || grid->update_enable)
            continue;

        Eigen::Vector3d grid_center = grid->center;

        double trace = grid->plane->cov.block<3, 3>(0, 0).trace();
        if (trace >= 0.25)
            trace = 0.25;
        trace = trace * (1.0 / 0.25);
        trace = pow(trace, 0.2);
        uint8_t r, g, b;
        mapJet(trace, 0, 1, r, g, b);
        Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
        double alpha = 0.8;

        visualization_msgs::Marker plane;
        plane.header.frame_id = frame_id;
        plane.header.stamp = ros::Time().fromSec(timestamp);
        plane.ns = "plane";
        plane.id = count++;
        if (!grid->merged)
            plane.type = visualization_msgs::Marker::CYLINDER;
        else
            plane.type = visualization_msgs::Marker::CUBE;
        plane.action = visualization_msgs::Marker::ADD;
        plane.pose.position.x = grid_center[0];
        plane.pose.position.y = grid_center[1];
        plane.pose.position.z = grid_center[2];
        geometry_msgs::Quaternion q;
        calcVectQuation(grid->plane->norm, q);
        plane.pose.orientation = q;
        plane.scale.x = voxel_size;
        plane.scale.y = voxel_size;
        plane.scale.z = 0.01;
        plane.color.a = alpha;
        plane.color.r = plane_rgb[0];
        plane.color.g = plane_rgb[1];
        plane.color.b = plane_rgb[2];
        plane.lifetime = ros::Duration();
        voxel_plane.markers.push_back(plane);
    }
    return voxel_plane;
}