cmake_minimum_required(VERSION 3.0.2)
project(std_matcher)

add_compile_options(-std=c++14)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3" )
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
  pcl_ros
  interface
)

find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES std_matcher
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

set(SRC_LIST src/std_manager/utils.cpp)
set(STD_SRC_LIST src/std_manager/voxel_map.cpp)

add_executable(temp_node src/temp_node.cpp ${SRC_LIST})
target_link_libraries(temp_node
  ${catkin_LIBRARIES}
)

add_executable(demo_node0 src/demo_node0.cpp  ${SRC_LIST} ${STD_SRC_LIST})

target_link_libraries(demo_node0
  ${catkin_LIBRARIES}
)
