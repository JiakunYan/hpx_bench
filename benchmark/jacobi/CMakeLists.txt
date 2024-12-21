# Copyright (c) 2012 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

# add_subdirectory(jacobi_component)

# add_executable(jacobi jacobi.cpp) add_subdirectory(jacobi_component)
# target_link_libraries(jacobi PRIVATE HPX::hpx) install( TARGETS jacobi
# DESTINATION ${CMAKE_INSTALL_BINDIR})

add_subdirectory(jacobi_component)

set(jacobi_sources jacobi.cpp)

set(jacobi_FLAGS COMPONENT_DEPENDENCIES iostreams jacobi)

# add jacobi executable
add_hpx_executable(
  jacobi_simple
  INTERNAL_FLAGS
  MODULE
  jacobi
  SOURCES
  ${jacobi_sources}
  ${jacobi_FLAGS}
  FOLDER
  "Examples/Jacobi")

set_target_properties(jacobi_simple PROPERTIES OUTPUT_NAME "hpxb_jacobi")
install(TARGETS jacobi_simple RUNTIME)
