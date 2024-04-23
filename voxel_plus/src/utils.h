#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "map_builder/voxel_map.h"
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/MarkerArray.h>

/**
 * msg: input
 * out: output lidar
 * filter_num: sample rate
 * blind: filter out near range
 */
void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double range_min, double range_max);

/**
 *
 */
sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp);

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);

void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

void calcVectQuation(const Eigen::Vector3d &norm, geometry_msgs::Quaternion &q);

visualization_msgs::MarkerArray voxel2MarkerArray(std::shared_ptr<lio::VoxelMap> map, const std::string &frame_id, const double &timestamp, int max_capacity = 1000000, double voxel_size = 0.2);