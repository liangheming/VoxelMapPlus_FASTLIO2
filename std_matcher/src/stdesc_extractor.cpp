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
    double res = std::stod(argv[3]);
    if (res > 0.0)
    {
        filter.setLeafSize(res, res, res);
        filter.setInputCloud(cloud);
        filter.filter(*cloud);
    }
    writer.writeBinaryCompressed(argv[2], *cloud);
    return 0;
}