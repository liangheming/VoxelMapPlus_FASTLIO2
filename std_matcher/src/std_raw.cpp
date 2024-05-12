#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "std_manager/STDesc.h"

int main(int argc, char **argv)
{

    // std::cout << "voxel size:" << config_setting.voxel_size_ << std::endl;
    // std::cout << "voxel_init_num:" << config_setting.voxel_init_num_ << std::endl;
    // std::cout << "maximum_corner_num:" << config_setting.maximum_corner_num_ << std::endl;
    // std::cout << "plane_merge_normal_thre:" << config_setting.plane_merge_normal_thre_ << std::endl;
    // std::cout << "descriptor_near_num:" << config_setting.descriptor_near_num_ << std::endl;
    // std::cout << "descriptor_min_len:" << config_setting.descriptor_min_len_ << std::endl;
    // std::cout << "descriptor_max_len:" << config_setting.descriptor_max_len_ << std::endl;
    // std::cout << "proj_image_resolution:" << config_setting.proj_image_resolution_ << std::endl;
    // std::cout << "proj_dis_min:" << config_setting.proj_dis_min_ << std::endl;
    // std::cout << "proj_dis_max:" << config_setting.proj_dis_max_ << std::endl;
    // std::cout << "corner_thre:" << config_setting.corner_thre_ << std::endl;
    // std::cout << "non_max_suppression_radius:" << config_setting.non_max_suppression_radius_ << std::endl;
    ConfigSetting config_setting;
    STDescManager std_manager(config_setting);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PCDReader reader;
    reader.read(argv[1], *cloud);
    {
        std::cout << "point size1:" << cloud->size() << std::endl;
        std::vector<STDesc> stds_vec;
        std_manager.GenerateSTDescs(cloud, stds_vec);
        std::cout << "feature size1: " << stds_vec.size() << std::endl;
    }

    cloud->clear();

    reader.read(argv[2], *cloud);
    {
        std::cout << "point size2:" << cloud->size() << std::endl;
        std::vector<STDesc> stds_vec;
        std_manager.GenerateSTDescs(cloud, stds_vec);
        std::cout << "feature size2: " << stds_vec.size() << std::endl;
    }

    return 0;
}