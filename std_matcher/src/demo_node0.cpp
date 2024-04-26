#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "std_manager/voxel_map.h"
#include <pcl/filters/voxel_grid.h>

int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    reader.read("/home/zhouzhou/temp/saved_pcd/1490.pcd", *cloud);
    pcl::VoxelGrid<pcl::PointXYZINormal> filter;
    filter.setLeafSize(0.1, 0.1, 0.1);
    filter.setInputCloud(cloud);
    filter.filter(*cloud);

    stdes::VoxelMap voxel_map(1.0, 0.005, 10);

    voxel_map.build(cloud);
    voxel_map.mergePlanes();

    std::cout << "plane voxel size: " << voxel_map.plane_voxels.size() << std::endl;
    std::cout << "total voxel size: " << voxel_map.voxels.size() << std::endl;
    std::cout << "merged plane size: " << voxel_map.planes.size() << std::endl;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> visual_clouds = voxel_map.coloredPlaneCloud();
    for (int i = 0; i < visual_clouds.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored_cloud = visual_clouds[i];
        if (colored_cloud->size() > 0)
            writer.write("/home/zhouzhou/temp/single_pcd/" + std::to_string(i) + ".pcd", *colored_cloud);
    }

    return 0;
}