add_library(test_tls_lib SHARED)
target_include_directories(test_tls_lib PUBLIC lib)
target_sources(test_tls_lib PRIVATE lib/test_tls_lib.cpp)

add_executable(test_tls test_tls.cpp)
target_link_libraries(test_tls PRIVATE test_tls_lib)

install(
        TARGETS test_tls
        DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
        TARGETS test_tls_lib
        DESTINATION ${CMAKE_INSTALL_LIBDIR})