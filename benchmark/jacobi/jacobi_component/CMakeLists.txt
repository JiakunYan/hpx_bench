# Copyright (c) 2012 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(
  jacobi
  INTERNAL_FLAGS
  NOEXPORT
  AUTOGLOB
  INSTALL_COMPONENT
  examples
  SOURCE_GLOB
  "*.c*"
  HEADER_GLOB
  "*.h*"
  ${jacobi_component_FLAGS}
  FOLDER
  "Examples/Jacobi"
  COMPONENT_DEPENDENCIES
  iostreams)

# cmake_policy(PUSH) cmake_policy(SET CMP0076 NEW) target_sources(jacobi PRIVATE
# grid.cpp jacobi_component.cpp row.cpp stencil_iterator.cpp server/row.cpp
# server/solver.cpp stencil_iterator.cpp) cmake_policy(POP)
# target_include_directories(jacobi PRIVATE . server)
