find_package(FFTW3 REQUIRED)

add_executable(fft_loop fft_hpx_loop.cpp)
target_link_libraries(fft_loop PRIVATE HPX::hpx HPX::iostreams_component
                                       ${FFTW3_LIBRARIES})
target_include_directories(fft_loop PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_directories(fft_loop PRIVATE ${FFTW3_LIBRARY_DIRS})
set_target_properties(fft_loop PROPERTIES OUTPUT_NAME "hpxb_fft_loop")
install(TARGETS fft_loop RUNTIME)

add_executable(fft_task fft_hpx_task_agas.cpp)
target_link_libraries(fft_task PRIVATE HPX::hpx HPX::iostreams_component
                                       ${FFTW3_LIBRARIES})
target_include_directories(fft_task PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_directories(fft_task PRIVATE ${FFTW3_LIBRARY_DIRS})
set_target_properties(fft_task PROPERTIES OUTPUT_NAME "hpxb_fft_task")
install(TARGETS fft_task RUNTIME)
