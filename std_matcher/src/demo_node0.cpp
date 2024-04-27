#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "std_manager/voxel_map.h"
#include <pcl/filters/voxel_grid.h>
#include <chrono>

int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    reader.read("/home/zhouzhou/temp/saved_pcd/10.pcd", *cloud);
    pcl::VoxelGrid<pcl::PointXYZINormal> filter;
    filter.setLeafSize(0.1, 0.1, 0.1);
    filter.setInputCloud(cloud);
    filter.filter(*cloud);

    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<stdes::VoxelMap> voxel_map = std::make_shared<stdes::VoxelMap>(1.0, 0.005, 10);
    stdes::STDExtractor extractor(voxel_map);
    std::vector<stdes::STDDescriptor> descs;
    extractor.extract(cloud, 1, descs);
    std::cout << descs.size() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;

    return 0;
}