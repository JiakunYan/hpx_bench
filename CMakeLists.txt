cmake_minimum_required(VERSION 3.12)
project(hpx_bench)

set(CMAKE_CXX_STANDARD 17)

find_package(HPX REQUIRED)

add_executable(pingpong_performance2 pingpong_performance2.cpp)
target_link_libraries(pingpong_performance2 PRIVATE HPX::hpx)

add_executable(test_barrier test_barrier.cpp)
target_link_libraries(test_barrier PRIVATE HPX::hpx)

add_executable(test_kernel test_kernel.cpp)
target_link_libraries(test_kernel)

install(
        TARGETS pingpong_performance2 test_kernel
        DESTINATION ${CMAKE_INSTALL_BINDIR})

add_subdirectory(test_tls)