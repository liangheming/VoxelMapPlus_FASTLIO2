cmake_minimum_required(VERSION 3.0.2)
project(voxel_plus)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
add_definitions(-DMP_EN)
add_definitions(-DMP_PROC_NUM=1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf2_ros
  pcl_ros
  nav_msgs
  sensor_msgs
  interface
  livox_ros_driver2
)

find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Sophus REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

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
             src/map_builder/voxel_map.cpp
             src/map_builder/lio_builder.cpp
             src/map_builder/voxel_map.cpp
             )

add_executable(lio_node src/lio_node.cpp ${SRC_LIST})


target_link_libraries(lio_node
  ${catkin_LIBRARIES}
)
