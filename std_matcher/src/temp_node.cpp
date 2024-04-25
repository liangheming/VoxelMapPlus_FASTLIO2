#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include "std_manager/utils.h"
#include "interface/PointCloudWithOdom.h"
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

pcl::PCDWriter writer;
std::string pcd_dir = "/home/zhouzhou/temp/saved_pcd/";
int sub_map_num = 10;
uint64_t count = 0;

pcl::PointCloud<pcl::PointXYZINormal>::Ptr acc_cloud;

Eigen::Quaterniond last_key_q;

Eigen::Vector3d last_key_p;
double p_thresh = 0.2, rad_thresh = 0.25;
bool need_save = false;

bool isKeyFrame(const Eigen::Quaterniond &q, const Eigen::Vector3d &p)
{
    Eigen::Quaterniond delta_q = last_key_q.inverse() * q;
    Eigen::Vector3d delta_p = p - last_key_p;
    Eigen::Matrix3d delta_r = delta_q.toRotationMatrix();
    Eigen::Vector3d rpy = rotate2rpy(delta_r);

    if (delta_p.norm() > p_thresh || rpy(0) > rad_thresh || rpy(1) > rad_thresh || rpy(2) > rad_thresh)
        return true;
    else
        return false;
}

void pwoCB(const interface::PointCloudWithOdom::ConstPtr msg)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::fromROSMsg(msg->cloud, *cloud);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZINormal>);

    Eigen::Quaterniond c_q;
    Eigen::Vector3d c_p;

    c_p.x() = msg->pose.pose.position.x;
    c_p.y() = msg->pose.pose.position.y;
    c_p.z() = msg->pose.pose.position.z;
    c_q.x() = msg->pose.pose.orientation.x;
    c_q.y() = msg->pose.pose.orientation.y;
    c_q.z() = msg->pose.pose.orientation.z;
    c_q.w() = msg->pose.pose.orientation.w;

    pcl::transformPointCloud(*cloud, *cloud_world, c_p.cast<float>(), c_q.cast<float>());

    if (count % sub_map_num == 0)
    {
        acc_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        *acc_cloud += *cloud_world;
        if (count == 0 || isKeyFrame(c_q, c_p))
        {
            last_key_p = c_p;
            last_key_q = c_q;
            need_save = true;
        }
    }
    else
    {
        *acc_cloud += *cloud_world;
    }
    count++;

    if (count % sub_map_num == 0 && need_save)
    {
        std::string file_name = pcd_dir + std::to_string(count) + ".pcd";
        ROS_INFO("key frame added! %lu ", count);
        writer.write(file_name, *acc_cloud);
        need_save = false;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "temp_node");
    ros::NodeHandle nh("~");
    ros::Subscriber sub = nh.subscribe<interface::PointCloudWithOdom>("/lio_node/cloud_with_odom", 1000, pwoCB);
    ros::spin();
    return 0;
}