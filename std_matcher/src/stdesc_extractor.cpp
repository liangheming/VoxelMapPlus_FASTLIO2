#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "std_manager/voxel_map.h"
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    reader.read(argv[1], *cloud);
    pcl::VoxelGrid<pcl::PointXYZINormal> filter;
    filter.setLeafSize(0.1, 0.1, 0.1);
    filter.setInputCloud(cloud);
    filter.filter(*cloud);

    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<stdes::VoxelMap> voxel_map = std::make_shared<stdes::VoxelMap>(1.0, 0.01, 10);
    stdes::STDExtractor extractor(voxel_map);
    extractor.nms_3d_range = 2.0;
    extractor.min_dis_threshold = 2.0;
    extractor.descriptor_near_num = 15;
    std::vector<stdes::STDDescriptor> descs;
    extractor.extract(cloud, 1, descs);
    std::cout << descs.size() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr visualize_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    for (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &p : extractor.voxel_map->coloredPlaneCloud(false))
    {
        *visualize_cloud += *p;
    }
    return 0;
}