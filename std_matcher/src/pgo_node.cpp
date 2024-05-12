#include <queue>
#include <chrono>
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
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

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
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_buffer;
    std::queue<Eigen::Affine3d> pose_buffer;
    std::queue<double> time_buffer;

    double current_time;
    Eigen::Affine3d current_pose;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr key_cloud;

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
        loadParams();
        initSubScribers();
        initPublishers();
        initSAM();
        main_loop = nh.createTimer(ros::Duration(0.05), &PGONode::mainLoopCB, this);
        std_manager = std::make_shared<std_desc::STDManager>(std_config);
    }

    void loadParams()
    {
        nh.param<std::string>("map_frame", node_config.map_frame, "map");
        nh.param<std::string>("odom_cloud_topic", node_config.odom_cloud_topic, "/lio_node/cloud_with_odom");
        nh.param<double>("ds_size", node_config.ds_size, 0.25);
        nh.param<int>("sub_frame_num", node_config.sub_frame_num, 10);

        nh.param<double>("voxel_size", std_config.voxel_size, 1.0);
        nh.param<int>("voxel_min_point", std_config.voxel_min_point, 10);
        nh.param<double>("voxel_plane_thresh", std_config.voxel_plane_thresh, 0.01);
        nh.param<double>("norm_merge_thresh", std_config.norm_merge_thresh, 0.1);

        nh.param<double>("proj_2d_resolution", std_config.proj_2d_resolution, 0.25);
        nh.param<double>("proj_min_dis", std_config.proj_min_dis, 0.0001);
        nh.param<double>("proj_max_dis", std_config.proj_max_dis, 5.0);

        nh.param<int>("nms_2d_range", std_config.nms_2d_range, 5);
        nh.param<double>("nms_3d_range", std_config.nms_3d_range, 2.0);
        nh.param<double>("corner_thresh", std_config.corner_thresh, 10.0);
        nh.param<int>("max_corner_num", std_config.max_corner_num, 100);
        nh.param<double>("min_side_len", std_config.min_side_len, 2.0);
        nh.param<double>("max_side_len", std_config.proj_max_dis, 30.0);

        nh.param<int>("desc_search_range", std_config.desc_search_range, 15);

        nh.param<double>("side_resolution", std_config.side_resolution, 0.2);
        nh.param<double>("rough_dis_threshold", std_config.rough_dis_threshold, 0.03);
        nh.param<int>("skip_near_num", std_config.skip_near_num, 50);
        nh.param<int>("candidate_num", std_config.candidate_num, 50);
        nh.param<double>("vertex_diff_threshold", std_config.vertex_diff_threshold, 0.7);
        nh.param<double>("verify_dis_thresh", std_config.verify_dis_thresh, 3.0);
        nh.param<double>("geo_verify_dis_thresh", std_config.geo_verify_dis_thresh, 0.3);
        nh.param<double>("icp_thresh", std_config.icp_thresh, 0.5);

        nh.param<double>("iter_eps", std_config.iter_eps, 0.001);
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

    void publishPath(ros::Publisher &pub, std::vector<Eigen::Affine3d> &pose_list)
    {
        if (pub.getNumSubscribers() < 1)
            return;
        nav_msgs::Path path;
        for (int i = 0; i < pose_list.size(); i++)
        {
            geometry_msgs::PoseStamped msg_pose;
            msg_pose.pose.position.x = pose_list[i].translation()[0];
            msg_pose.pose.position.y = pose_list[i].translation()[1];
            msg_pose.pose.position.z = pose_list[i].translation()[2];
            Eigen::Quaterniond pose_q(pose_list[i].rotation());
            msg_pose.header.frame_id = node_config.map_frame;
            msg_pose.pose.orientation.x = pose_q.x();
            msg_pose.pose.orientation.y = pose_q.y();
            msg_pose.pose.orientation.z = pose_q.z();
            msg_pose.pose.orientation.w = pose_q.w();
            path.poses.push_back(msg_pose);
        }

        path.header.stamp = ros::Time().fromSec(data_group.current_time);
        path.header.frame_id = node_config.map_frame;
        pub.publish(path);
    }

    void publishLoopConstraints()
    {
        if (loop_contraints_pub.getNumSubscribers() < 1)
            return;
        if (data_group.loop_container.size() == 0)
            return;
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker_node;
        marker_node.header.frame_id = node_config.map_frame;
        marker_node.action = visualization_msgs::Marker::ADD;
        marker_node.type = visualization_msgs::Marker::SPHERE_LIST;
        marker_node.ns = "loop_nodes";
        marker_node.id = 0;
        marker_node.pose.orientation.w = 1;
        marker_node.scale.x = 0.3;
        marker_node.scale.y = 0.3;
        marker_node.scale.z = 0.3;
        marker_node.color.r = 0;
        marker_node.color.g = 0.8;
        marker_node.color.b = 1;
        marker_node.color.a = 1;

        visualization_msgs::Marker marker_edge;
        marker_edge.header.frame_id = node_config.map_frame;
        marker_edge.action = visualization_msgs::Marker::ADD;
        marker_edge.type = visualization_msgs::Marker::LINE_LIST;
        marker_edge.ns = "loop_edges";
        marker_edge.id = 1;
        marker_edge.pose.orientation.w = 1;
        marker_edge.scale.x = 0.1;
        marker_edge.color.r = 0.9;
        marker_edge.color.g = 0.9;
        marker_edge.color.b = 0;
        marker_edge.color.a = 1;

        for (auto it = data_group.loop_container.begin(); it != data_group.loop_container.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
            p.x = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
        }

        marker_array.markers.push_back(marker_node);
        marker_array.markers.push_back(marker_edge);
        loop_contraints_pub.publish(marker_array);
    }

    void initPublishers()
    {
        path_pub = nh.advertise<nav_msgs::Path>("origin_path", 10000);
        correct_path_pub = nh.advertise<nav_msgs::Path>("correct_path", 10000);
        loop_contraints_pub = nh.advertise<visualization_msgs::MarkerArray>("loop_contriants", 10);
    }

    void odomCloudCB(const interface::PointCloudWithOdom::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(msg->cloud, *cloud);
        node_config.local_frame = msg->header.frame_id;
        data_group.cloud_buffer.push(cloud);
        Eigen::Quaterniond rotation(msg->pose.pose.orientation.w,
                                    msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y,
                                    msg->pose.pose.orientation.z);
        Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = rotation.toRotationMatrix();
        pose.translation() = translation;
        data_group.pose_buffer.push(pose);
        data_group.time_buffer.push(msg->header.stamp.toSec());
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
                data_group.cloud_buffer.pop();
                data_group.pose_buffer.pop();
                data_group.time_buffer.pop();
            }
        }
        pcl::transformPointCloud(*data_group.current_cloud, *data_group.current_cloud, data_group.current_pose);

        std_desc::voxelFilter(data_group.current_cloud, node_config.ds_size);

        if (data_group.key_cloud == nullptr)
            data_group.key_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        *data_group.key_cloud += *data_group.current_cloud;

        Eigen::Affine3d pose = data_group.current_pose;
        // pose.linear() = data_group.current_pose.first;
        // pose.translation() = data_group.current_pose.second;
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
            // if (data_group.cloud_idx == 660)
            // {
            //     pcl::PCDWriter writer;
            //     writer.write("/home/zhouzhou/temp/660_new.pcd", *data_group.key_cloud);
            // }
            std_desc::STDFeature feature = std_manager->extract(data_group.key_cloud);
            ROS_INFO("ID: %lu  FEATRUE SIZE: %lu CLOUD SIZE: %lu", data_group.cloud_idx, feature.descs.size(), data_group.key_cloud->size());
            std_desc::LoopResult result;

            int64_t duration;
            if (data_group.cloud_idx > std_config.skip_near_num)
            {
                auto start = std::chrono::high_resolution_clock::now();
                result = std_manager->searchLoop(feature);
                auto end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }

            std_manager->insert(feature);
            if (result.valid)
            {

                // 这里得到是新旧世界坐标系下的差值 T_old_new;
                ROS_WARN("FIND MATCHED LOOP! CURRENT_ID: %lu, LOOP_ID: %lu MATCH SCORE: %.4f, TIME COST: %lu ms", feature.id, result.match_id, result.match_score, duration);
                double score = std_manager->verifyGeoPlaneICP(feature.cloud, std_manager->cloud_vec[result.match_id], result.rotation, result.translation);
                data_group.has_loop_flag = true;
                data_group.loop_container.emplace_back(result.match_id, feature.id);
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

        publishPath(path_pub, data_group.origin_pose_vec);

        publishPath(correct_path_pub, data_group.pose_vec);

        publishLoopConstraints();

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

    ros::Publisher correct_path_pub;
    ros::Publisher path_pub;
    ros::Publisher loop_contraints_pub;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pgo_node");
    PGONode pgo_node;
    ros::spin();
    return 0;
}