#include <iostream>
#include "std_manager/descriptor.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

int main(int argc, char **argv)
{
    std_desc::Config config;
    
    std_desc::STDManager manager(config);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PCDReader reader;
    reader.read(argv[1], *cloud);
    std::cout << "point size:" << cloud->size() << std::endl;
    std_desc::STDFeature feature = manager.extract(cloud);
    std::cout << "feature size2: " << feature.descs.size() << std::endl;

    cloud->clear();
    reader.read(argv[2], *cloud);
    std::cout << "point size:" << cloud->size() << std::endl;
    std_desc::STDFeature feature2 = manager.extract(cloud);
    std::cout << "feature size2: " << feature2.descs.size() << std::endl;

    return 0;
}