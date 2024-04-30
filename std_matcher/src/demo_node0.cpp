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
    extractor.nms_3d_range = 1.5;
    extractor.descriptor_near_num = 15;
    std::vector<stdes::STDDescriptor> descs;
    extractor.extract(cloud, 1, descs);
    std::cout << descs.size() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr visualize_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    for (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &p : extractor.voxel_map->coloredPlaneCloud(true))
    {
        *visualize_cloud += *p;
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGBA>(visualize_cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    for (int i = 0; i < descs.size(); i++)
    {
        stdes::STDDescriptor &desc_i = descs[i];
        pcl::PointXYZ p1, p2, p3;
        p1.x = desc_i.vertex_a.x();
        p1.y = desc_i.vertex_a.y();
        p1.z = desc_i.vertex_a.z();

        p2.x = desc_i.vertex_b.x();
        p2.y = desc_i.vertex_b.y();
        p2.z = desc_i.vertex_b.z();

        p3.x = desc_i.vertex_c.x();
        p3.y = desc_i.vertex_c.y();
        p3.z = desc_i.vertex_c.z();

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