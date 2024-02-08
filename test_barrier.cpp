//  Copyright (c) 2023 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/serialization.hpp>
#include <hpx/collectives/barrier.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

void run_test();
HPX_PLAIN_ACTION(run_test, run_test_action)

void run_test() {
    fprintf(stderr, "%d: run_test\n", hpx::get_locality_id());
    hpx::distributed::barrier::synchronize();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& b_arg)
{
    std::vector<hpx::future<void>> futs;
    for (auto l : hpx::find_all_localities()) {
        futs.emplace_back(hpx::async<run_test_action>(l));
    }
    hpx::wait_all(futs);

    hpx::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;
    po::options_description description("HPX test barrier");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    return hpx::init(argc, argv, init_args);
}

#endif
