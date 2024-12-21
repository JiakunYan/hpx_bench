# Copyright (c) 2011      Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

# add example executable
add_hpx_executable(
  hpcc_random_access
  INTERNAL_FLAGS
  MODULE
  hpcc_random_access
  SOURCES
  hpcc_random_access.cpp
  FOLDER
  "Examples/RandomMemoryAccess/hpcc_random_access")

set_target_properties(hpcc_random_access PROPERTIES OUTPUT_NAME "hpxb_raccess")
install(TARGETS hpcc_random_access RUNTIME)
