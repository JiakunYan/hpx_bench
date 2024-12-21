add_executable(pingpong pingpong.cpp)
target_link_libraries(pingpong PRIVATE HPX::hpx)
set_target_properties(pingpong PROPERTIES OUTPUT_NAME "hpxb_pingpong")
install(TARGETS pingpong RUNTIME)
