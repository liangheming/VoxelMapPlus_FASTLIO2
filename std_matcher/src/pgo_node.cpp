#include <queue>
#include <ros/ros.h>
#include "std_manager/descriptor.h"
#include "interface/PointCloudWithOdom.h"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

struct Config
{
    double ds_size = 0.25;
    int sub_frame_num = 10;

    std::string odom_cloud_topic = "/lio_node/cloud_with_odom";
};
struct DataGroup
{
    std::mutex buffer_mutex;
    std::deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> cloud_buffer;
    std::deque<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_buffer;

    std::pair<Eigen::Matrix3d, Eigen::Vector3d> current_pose;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr current_cloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr key_cloud;

    size_t cloud_idx = 0;
};

class PGONode
{
public:
    PGONode() : nh("~")
    {
        initSubScribers();
        main_loop = nh.createTimer(ros::Duration(0.05), &PGONode::mainLoopCB, this);
        std_manager = std::make_shared<std_desc::STDManager>(std_config);
    }

    void initSubScribers()
    {
        odom_cloud_sub = nh.subscribe(node_config.odom_cloud_topic, 100, &PGONode::odomCloudCB, this);
    }

    void odomCloudCB(const interface::PointCloudWithOdom::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::fromROSMsg(msg->cloud, *cloud);
        data_group.cloud_buffer.push_back(cloud);
        Eigen::Quaterniond rotation(msg->pose.pose.orientation.w,
                                    msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y,
                                    msg->pose.pose.orientation.z);
        Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        data_group.pose_buffer.emplace_back(rotation.toRotationMatrix(), translation);
    }

    void mainLoopCB(const ros::TimerEvent &e)
    {
        {
            std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
            if (data_group.cloud_buffer.size() < 1)
                return;
            data_group.current_cloud = data_group.cloud_buffer.front();
            data_group.current_pose = data_group.pose_buffer.front();
            while (!data_group.cloud_buffer.empty())
            {
                data_group.cloud_buffer.pop_front();
                data_group.pose_buffer.pop_front();
            }
        }
        pcl::transformPointCloud(*data_group.current_cloud,
                                 *data_group.current_cloud,
                                 data_group.current_pose.second.cast<float>(),
                                 Eigen::Quaternionf(data_group.current_pose.first.cast<float>()));

        std_desc::voxelFilter(data_group.current_cloud, node_config.ds_size);
        if (data_group.key_cloud == nullptr)
            data_group.key_cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        *data_group.key_cloud += *data_group.current_cloud;

        if (data_group.cloud_idx % node_config.sub_frame_num == 0 && data_group.cloud_idx != 0)
        {
            std_desc::STDFeature feature = std_manager->extract(data_group.key_cloud);

            std_desc::LoopResult result;
            if (data_group.cloud_idx > std_config.skip_near_num)
            {
                result = std_manager->searchLoop(feature);
            }

            std_manager->insert(feature);
            ROS_INFO("ADD KEY FRAME DO SEARCH!");
            if (result.valid)
            {
                // 这里得到是新旧世界坐标系下的差值 T_old_new;
                ROS_WARN("FIND MATCHED LOOP! current_id: %lu, loop_id: %lu match_score: %.4f", feature.id, result.match_id, result.match_score);
                std::cout << "before icp " << result.translation.transpose() << std::endl;
                std::cout << result.rotation << std::endl;
                double score = std_manager->verifyGeoPlaneICP(feature.cloud, std_manager->cloud_vec[result.match_id], result.rotation, result.translation);
                std::cout << "after icp " << result.translation.transpose() << std::endl;
                std::cout << result.rotation << std::endl;
                std::cout << "res sum " << score << std::endl;
            }
            data_group.key_cloud->clear();
        }
        data_group.cloud_idx++;
    }

public:
    ros::NodeHandle nh;
    Config node_config;
    tf2_ros::TransformBroadcaster br;
    ros::Timer main_loop;
    ros::Subscriber odom_cloud_sub;
    DataGroup data_group;
    std_desc::Config std_config;
    std::shared_ptr<std_desc::STDManager> std_manager;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pgo_node");
    PGONode pgo_node;
    ros::spin();
    return 0;
}