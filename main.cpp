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

///////////////////////////////////////////////////////////////////////////////
const std::size_t nbytes_default = 8;
const std::size_t nsteps_default = 1;
const std::size_t nchains_default = 10000;
const std::size_t inject_rate_default = 0;
const std::size_t batch_size_default = 10;
const std::size_t nwarmups_default = 1;
const std::size_t niters_default = 1;
const std::size_t is_single_source_default = false;

struct config_t
{
    size_t nbytes;
    size_t nsteps;
    size_t nchains;
    size_t inject_rate;
    size_t batch_size;
    size_t nwarmups;
    size_t niters;
    bool is_single_source;
    bool verbose;
} config;
size_t nchains_per_rank;

///////////////////////////////////////////////////////////////////////////////

void set_config(config_t config_);
HPX_PLAIN_ACTION(set_config, set_config_action)

void run_bench();
HPX_PLAIN_ACTION(run_bench, run_bench_action)

void on_inject(hpx::id_type to);
HPX_PLAIN_ACTION(on_inject, on_inject_action)

void on_recv(hpx::id_type to, std::vector<char> const& in, std::size_t counter);
HPX_PLAIN_ACTION(on_recv, on_recv_action)

void on_done();
HPX_PLAIN_ACTION(on_done, on_done_action)

void set_config(config_t config_)
{
    config = config_;
}

void on_inject(hpx::id_type to)
{
    hpx::chrono::high_resolution_timer timer;
    for (size_t i = 0; i < config.batch_size; ++i)
    {
        while (config.inject_rate > 0 &&
               static_cast<double>(i) / timer.elapsed() >
                       static_cast<double>(config.inject_rate))
        {
            hpx::this_thread::yield();
        }
        std::vector<char> data(config.nbytes, 'a');
        hpx::post<on_recv_action>(to, hpx::find_here(), data, config.nsteps);
    }
}

std::atomic<size_t> done_counter(0);

void on_recv(hpx::id_type to, std::vector<char> const& in, std::size_t counter)
{
    // received vector in
    if (--counter == 0)
    {
        size_t result = done_counter.fetch_add(1, std::memory_order_relaxed);
        if (result + 1 == nchains_per_rank)
        {
            if (config.is_single_source)
                hpx::post<on_done_action>(hpx::find_root_locality());
            else
                hpx::post<on_done_action>(hpx::find_here());

            done_counter = 0;
        }
        return;
    }

    // send it to remote locality (to)
    std::vector<char> data(in);
    hpx::post<on_recv_action>(to, hpx::find_here(), std::move(data), counter);
}

hpx::counting_semaphore_var<> semaphore;

void on_done()
{
    semaphore.signal();
}

void run_bench()
{
    auto localities = hpx::find_all_localities();
    auto rank = hpx::get_locality_id();
    auto nranks = hpx::get_num_localities().get();
    hpx::id_type peer;
    auto peers = hpx::find_remote_localities();
    if (peers.size() == 0) {
        peer = hpx::find_here();
    } else {
        HPX_ASSERT(peers.size() == 1);
        peer = peers[0];
    }
    double inject_time = 0;
    double time = 0;
    nchains_per_rank = (config.is_single_source)?
                                                 config.nchains : config.nchains / nranks;
    for (size_t j = 0; j < config.nwarmups + config.niters; ++j)
    {
        if (!config.is_single_source)
            hpx::distributed::barrier::synchronize();
        hpx::chrono::high_resolution_timer timer_total;
        for (size_t i = 0; i < nchains_per_rank; i += config.batch_size)
        {
            while (config.inject_rate > 0 &&
                   static_cast<double>(i) / timer_total.elapsed() >
                           static_cast<double>(config.inject_rate))
            {
                continue;
            }
            hpx::post<on_inject_action>(
                    hpx::find_here(), peer);
        }

        if (j >= config.nwarmups)
        {
            if (!config.is_single_source)
                hpx::distributed::barrier::synchronize();
            inject_time += timer_total.elapsed();
        }

        semaphore.wait();
        if (j >= config.nwarmups)
        {
            if (!config.is_single_source)
                hpx::distributed::barrier::synchronize();
            time += timer_total.elapsed();
        }
    }

    if (rank == 0)
    {
        double achieved_inject_rate =
                static_cast<double>(config.nchains * config.niters) / inject_time /
                1e3;
        double latency =
                time * 1e6 / static_cast<double>(config.nsteps * config.niters);
        double msg_rate = static_cast<double>(
                                  config.nsteps * config.nchains * config.niters) /
                          time / 1e3;
        double bandwidth = static_cast<double>(config.nbytes * config.nsteps *
                                               config.nchains * config.niters) /
                           time / 1e6;
        if (config.verbose)
        {
            std::cout << "[hpx_pingpong]" << std::endl
                      << "total_time(secs)=" << time << std::endl
                      << "nwarmups=" << config.nwarmups << std::endl
                      << "niters=" << config.niters << std::endl
                      << "nbytes=" << config.nbytes << std::endl
                      << "nchains=" << config.nchains << std::endl
                      << "is_single_source=" << config.is_single_source << std::endl
                      << "latency(us)=" << latency << std::endl
                      << "inject_rate(K/s)=" << achieved_inject_rate
                      << std::endl
                      << "msg_rate(K/s)=" << msg_rate << std::endl
                      << "bandwidth(MB/s)=" << bandwidth << std::endl
                      << "localities=" << localities.size() << std::endl
                      << "nsteps=" << config.nsteps << std::endl;
        }
        else
        {
            std::cout << "[hpx_pingpong]"
                      << ":total_time(secs)=" << time << ":nbytes=" << config.nbytes
                      << ":nwarmups=" << config.nwarmups
                      << ":niters=" << config.niters
                      << ":nchains=" << config.nchains
                      << ":is_single_source=" << config.is_single_source
                      << ":latency(us)=" << latency
                      << ":inject_rate(K/s)=" << achieved_inject_rate
                      << ":msg_rate(M/s)=" << msg_rate
                      << ":bandwidth(MB/s)=" << bandwidth
                      << ":localities=" << localities.size()
                      << ":nsteps=" << config.nsteps << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& b_arg)
{
    config.nbytes = b_arg["nbytes"].as<std::size_t>();
    config.nsteps = b_arg["nsteps"].as<std::size_t>();
    config.verbose = b_arg["verbose"].as<bool>();
    config.nchains = b_arg["nchains"].as<std::size_t>();
    config.inject_rate = b_arg["inject-rate"].as<std::size_t>();
    config.batch_size = b_arg["batch-size"].as<std::size_t>();
    config.nwarmups = b_arg["nwarmups"].as<std::size_t>();
    config.niters = b_arg["niters"].as<std::size_t>();
    config.is_single_source = b_arg["is-single-source"].as<bool>();

    if (config.nsteps == 0)
    {
        std::cout << "nsteps is 0!" << std::endl;
        return 0;
    }

    if (config.nchains == 0)
    {
        std::cout << "nchains is 0!" << std::endl;
        return 0;
    }

    if (hpx::get_num_localities().get() > 2) {
        std::cout << "Too many localities!" << std::endl;
        return 0;
    }

    for (auto peer : hpx::find_remote_localities())
    {
        set_config_action act;
        act(peer, config);
    }

    if (hpx::get_locality_id() == 0 || !config.is_single_source) {
        run_bench();
    }

    hpx::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;
    po::options_description description("HPX pingpong example");

    description.add_options()
            ("nbytes", po::value<std::size_t>()->default_value(nbytes_default),
             "number of elements (doubles) to send/receive (integer)")
                    ("nsteps", po::value<std::size_t>()->default_value(nsteps_default),
                     "number of ping-pong iterations")
                            ("nchains", po::value<std::size_t>()->default_value(nchains_default),
                             "nchains size of ping-pong")
                                    ("inject-rate",
                                     po::value<std::size_t>()->default_value(inject_rate_default),
                                     "the rate of injecting the first message of ping-pong")
                                            ("batch-size",
                                             po::value<std::size_t>()->default_value(batch_size_default),
                                             "the number of messages to inject per inject thread")
                                                    ("nwarmups",
                                                     po::value<std::size_t>()->default_value(nwarmups_default),
                                                     "the iteration count of warmup runs")
                                                            ("niters",
                                                             po::value<std::size_t>()->default_value(niters_default),
                                                             "the iteration count of measurement iterations.")
                                                                    ("is-single-source",
                                                                     po::value<bool>()->default_value(is_single_source_default),
                                                                     "Spawn all message chains from a single source or not")("verbose",
                                                                                                                             po::value<bool>()->default_value(true),
                                                                                                                             "verbosity of output,if false output is for awk");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    return hpx::init(argc, argv, init_args);
}

#endif
