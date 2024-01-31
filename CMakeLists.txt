cmake_minimum_required(VERSION 3.27)
project(hpx_bench)

set(CMAKE_CXX_STANDARD 17)

find_package(HPX REQUIRED)
add_executable(hpx_bench main.cpp)
target_link_libraries(hpx_bench PRIVATE HPX::hpx)