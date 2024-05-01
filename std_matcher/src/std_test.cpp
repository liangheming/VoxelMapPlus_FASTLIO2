#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include "std_manager/STDesc.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char **argv)
{
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    reader.read(argv[1], *cloud);
    
    pcl::VoxelGrid<pcl::PointXYZI> filter;
    filter.setLeafSize(0.1, 0.1, 0.1);
    filter.setInputCloud(cloud);
    filter.filter(*cloud);

    ConfigSetting config_setting;
    config_setting.plane_merge_normal_thre_ = 0.1;
    config_setting.voxel_init_num_ = 10;
    config_setting.voxel_size_ = 1.0;
    config_setting.plane_detection_thre_ = 0.01;
    config_setting.maximum_corner_num_ = 500;
    config_setting.proj_dis_min_ = 0.0;
    config_setting.proj_dis_max_ = 5.0;
    config_setting.proj_image_resolution_ = 0.25;
    config_setting.corner_thre_ = 10;
    config_setting.descriptor_near_num_ = 15;
    config_setting.descriptor_min_len_ = 2.0;
    config_setting.descriptor_max_len_ = 30.0;
    config_setting.non_max_suppression_radius_ = 2.0;
    config_setting.std_side_resolution_ = 0.2;

    std::shared_ptr<STDescManager> std_manager = std::make_shared<STDescManager>(config_setting);
    std::vector<STDesc> stds_vec;
    std_manager->GenerateSTDescs(cloud, stds_vec);
    std::cout << "descriptor_size: " << stds_vec.size() << std::endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZI>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);

    for (int i = 0; i < stds_vec.size(); i++)
    {
        STDesc&desc_i = stds_vec[i];
        pcl::PointXYZ p1, p2, p3;
        p1.x = desc_i.vertex_A_.x();
        p1.y = desc_i.vertex_A_.y();
        p1.z = desc_i.vertex_A_.z();

        p2.x = desc_i.vertex_B_.x();
        p2.y = desc_i.vertex_B_.y();
        p2.z = desc_i.vertex_B_.z();

        p3.x = desc_i.vertex_C_.x();
        p3.y = desc_i.vertex_C_.y();
        p3.z = desc_i.vertex_C_.z();

        viewer->addLine(p1, p2, 0., 1.0, 0., "tri" + std::to_string(i) + "_line1");
        viewer->addLine(p2, p3, 0., 1.0, 0., "tri" + std::to_string(i) + "_line2");
        viewer->addLine(p3, p1, 0., 1.0, 0., "tri" + std::to_string(i) + "_line3");
    }
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}