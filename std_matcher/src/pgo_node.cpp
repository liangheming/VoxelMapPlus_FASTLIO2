#include <queue>
#include <ros/ros.h>
#include "std_manager/descriptor.h"
#include "interface/PointCloudWithOdom.h"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <geometry_msgs/TransformStamped.h>

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

struct Config
{
    std::string map_frame = "map";
    std::string local_frame = "lidar";
    double ds_size = 0.25;
    int sub_frame_num = 10;
    std::string odom_cloud_topic = "/lio_node/cloud_with_odom";
};

struct DataGroup
{
    std::mutex buffer_mutex;
    std::deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> cloud_buffer;
    std::deque<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_buffer;
    std::deque<double> time_buffer;

    double current_time;
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> current_pose;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr current_cloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr key_cloud;

    size_t cloud_idx = 0;
    std::vector<Eigen::Affine3d> pose_vec;
    std::vector<Eigen::Affine3d> origin_pose_vec;
    std::vector<std::pair<int, int>> loop_container;

    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise;
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
    std::shared_ptr<gtsam::ISAM2> isam;

    bool has_loop_flag = false;
};

class PGONode
{
public:
    PGONode() : nh("~")
    {
        initSubScribers();
        initSAM();
        main_loop = nh.createTimer(ros::Duration(0.05), &PGONode::mainLoopCB, this);
        std_manager = std::make_shared<std_desc::STDManager>(std_config);
    }

    void initSAM()
    {
        data_group.odometryNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
        gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
            gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1), gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        data_group.isam = std::make_shared<gtsam::ISAM2>(parameters);
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
        node_config.local_frame = msg->header.frame_id;
        data_group.cloud_buffer.push_back(cloud);
        Eigen::Quaterniond rotation(msg->pose.pose.orientation.w,
                                    msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y,
                                    msg->pose.pose.orientation.z);
        Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        data_group.pose_buffer.emplace_back(rotation.toRotationMatrix(), translation);
        data_group.time_buffer.push_back(msg->header.stamp.toSec());
    }

    void mainLoopCB(const ros::TimerEvent &e)
    {
        {
            std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
            if (data_group.cloud_buffer.size() < 1)
                return;
            data_group.current_cloud = data_group.cloud_buffer.front();
            data_group.current_pose = data_group.pose_buffer.front();
            data_group.current_time = data_group.time_buffer.front();
            while (!data_group.cloud_buffer.empty())
            {
                data_group.cloud_buffer.pop_front();
                data_group.pose_buffer.pop_front();
                data_group.time_buffer.pop_front();
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

        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = data_group.current_pose.first;
        pose.translation() = data_group.current_pose.second;
        data_group.initial.insert(data_group.cloud_idx, gtsam::Pose3(pose.matrix()));
        if (!data_group.cloud_idx)
        {

            data_group.graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(pose.matrix()), data_group.odometryNoise));
        }
        else
        {
            auto prev_pose = gtsam::Pose3(data_group.origin_pose_vec[data_group.cloud_idx - 1].matrix());
            auto curr_pose = gtsam::Pose3(pose.matrix());
            data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(data_group.cloud_idx - 1, data_group.cloud_idx,
                                                                    prev_pose.between(curr_pose), data_group.odometryNoise));
        }

        data_group.origin_pose_vec.push_back(pose);
        data_group.pose_vec.push_back(pose);

        if (data_group.cloud_idx % node_config.sub_frame_num == 0 && data_group.cloud_idx != 0)
        {
            std_desc::STDFeature feature = std_manager->extract(data_group.key_cloud);

            std_desc::LoopResult result;
            if (data_group.cloud_idx > std_config.skip_near_num)
            {
                result = std_manager->searchLoop(feature);
            }

            std_manager->insert(feature);
            if (result.valid)
            {
                // 这里得到是新旧世界坐标系下的差值 T_old_new;
                ROS_WARN("FIND MATCHED LOOP! current_id: %lu, loop_id: %lu match_score: %.4f", feature.id, result.match_id, result.match_score);

                double score = std_manager->verifyGeoPlaneICP(feature.cloud, std_manager->cloud_vec[result.match_id], result.rotation, result.translation);
                // std::cout << "after icp " << result.translation.transpose() << std::endl;
                // std::cout << result.rotation << std::endl;
                // std::cout << "res sum " << score << std::endl;
                data_group.has_loop_flag = true;
                // 10 20
                for (size_t j = 1; j <= node_config.sub_frame_num; j++)
                {
                    // 当前帧
                    int src_frame = data_group.cloud_idx + j - node_config.sub_frame_num;
                    // 历史帧
                    int tar_frame = result.match_id * node_config.sub_frame_num + j;

                    Eigen::Affine3d delta_pose = Eigen::Affine3d::Identity();
                    delta_pose.linear() = result.rotation;
                    delta_pose.translation() = result.translation;

                    Eigen::Affine3d refined_src = delta_pose * data_group.origin_pose_vec[src_frame];
                    Eigen::Affine3d tar_pose = data_group.origin_pose_vec[tar_frame];

                    data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        tar_frame, src_frame,
                        gtsam::Pose3(tar_pose.matrix()).between(gtsam::Pose3(refined_src.matrix())),
                        data_group.robustLoopNoise));
                }
            }
            data_group.key_cloud->clear();
        }

        data_group.isam->update(data_group.graph, data_group.initial);
        data_group.isam->update();

        if (data_group.has_loop_flag)
        {
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
        }

        data_group.graph.resize(0);
        data_group.initial.clear();

        gtsam::Values curr_estimates = data_group.isam->calculateEstimate();

        assert(curr_estimates.size() == data_group.pose_vec.size());

        for (int i = 0; i < curr_estimates.size(); i++)
        {
            gtsam::Pose3 est = curr_estimates.at<gtsam::Pose3>(i);
            Eigen::Affine3d est_affine3d(est.matrix());
            data_group.pose_vec[i] = est_affine3d;
        }

        Eigen::Affine3d last_pose = data_group.pose_vec.back();

        Eigen::Affine3d frame_delta_pose = last_pose * data_group.origin_pose_vec.back().inverse();
        
        br.sendTransform(eigen2Transform(frame_delta_pose.linear(), frame_delta_pose.translation(), node_config.map_frame, node_config.local_frame, data_group.current_time));
        data_group.cloud_idx++;

        data_group.has_loop_flag = false;
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