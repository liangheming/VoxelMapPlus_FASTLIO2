cmake_minimum_required(VERSION 3.0.2)
project(voxel_plus)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf2_ros
  pcl_ros
  nav_msgs
  sensor_msgs
  livox_ros_driver2
)

find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES voxel_plus
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


set(SRC_LIST src/utils.cpp
             src/map_builder/commons.cpp
             src/map_builder/ieskf.cpp
             src/map_builder/lio_builder.cpp
             src/map_builder/voxel_map.cpp
             )

add_executable(lio_node src/lio_node.cpp ${SRC_LIST})


target_link_libraries(lio_node
  ${catkin_LIBRARIES}
)
