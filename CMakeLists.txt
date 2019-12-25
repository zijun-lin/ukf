cmake_minimum_required(VERSION 3.14)
project(UKF)

set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/include/eigen3)

set(FILES
        ukf.cpp ukf.h)

add_executable(UKF main.cpp ${FILES})
target_link_libraries(UKF m)