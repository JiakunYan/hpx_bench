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
const std::size_t nsteps_default = 1000;
const std::size_t nchains_default = 1024;
const std::size_t inject_rate_default = 0;
const std::size_t batch_size_default = 10;
const std::size_t nwarmups_default = 1;
const std::size_t niters_default = 1;
const std::size_t intensity_default = 0;
const std::size_t task_comp_time_us_default = 0;
const std::size_t intensity_eval_time_s_default = 1;
const bool is_single_source_default = false;
const bool verbose_default = true;

struct config_t {
  size_t nbytes;
  size_t nsteps;
  size_t nchains;
  size_t inject_rate;
  size_t batch_size;
  size_t nwarmups;
  size_t niters;
  size_t intensity;
  size_t task_comp_time_us;
  size_t intensity_eval_time_s;
  bool is_single_source;
  bool verbose;
} config;
size_t nchains_per_rank;
hpx::id_type here;
hpx::id_type peer;

///////////////////////////////////////////////////////////////////////////////

void set_config(config_t config_);
HPX_PLAIN_ACTION(set_config, set_config_action)

void run_bench();
HPX_PLAIN_ACTION(run_bench, run_bench_action)

void on_inject(int batch_size);
HPX_PLAIN_ACTION(on_inject, on_inject_action)

void on_recv(std::vector<double> const& in, std::size_t counter);
HPX_PLAIN_ACTION(on_recv, on_recv_action)

void on_done();
HPX_PLAIN_ACTION(on_done, on_done_action)

static inline void kernel_fma(double* T, size_t size, int intensity, int t)
{
  for (int j = 0; j < intensity; ++j)
    for (int i = 0; i < size; ++i) {
      //  T[i] = T[i] * t + j
      T[i] = fma(T[i], (double)t, (double)j);
    }
}

void evaluate_kernal()
{
  if (config.task_comp_time_us == 0 && config.intensity == 0) return;
  std::vector<double> data(config.nbytes / sizeof(double), 1);
  hpx::chrono::high_resolution_timer timer_eval;
  int intensity = 0;
  do {
    int j = ++intensity;
    for (int i = 0; i < data.size(); ++i) {
      //  T[i] = T[i] * t + j
      data[i] = fma(data[i], (double)j, (double)j);
    }
  } while (timer_eval.elapsed() <= config.intensity_eval_time_s);
  if (config.task_comp_time_us > 0) {
    config.intensity = (double)config.task_comp_time_us * (double)intensity /
                       (double)config.intensity_eval_time_s / 1e6;
  } else {
    config.task_comp_time_us = (double)config.intensity *
                               (double)config.intensity_eval_time_s * 1e6 /
                               (double)intensity;
  }
}

void set_config(config_t config_)
{
  config = config_;
  nchains_per_rank = (config.is_single_source)
                         ? config.nchains
                         : config.nchains / hpx::get_num_localities().get();
  here = hpx::find_here();
  auto peers = hpx::find_remote_localities();
  if (peers.size() == 0) {
    peer = here;
  } else {
    HPX_ASSERT(peers.size() == 1);
    peer = peers[0];
  }
}

void on_inject(int batch_size)
{
  hpx::chrono::high_resolution_timer timer;
  for (int i = 0; i < batch_size; ++i) {
    while (config.inject_rate > 0 &&
           static_cast<double>(i) / timer.elapsed() >
               static_cast<double>(config.inject_rate)) {
      hpx::this_thread::yield();
    }
    std::vector<double> data(config.nbytes / sizeof(double), 1);
    hpx::post<on_recv_action>(peer, data, config.nsteps);
  }
}

std::atomic<size_t> done_counter(0);

void on_recv(std::vector<double> const& in, std::size_t counter)
{
  //    fprintf(stderr, "%d: on_recv %lu\n", hpx::get_locality_id(), counter);
  std::vector<double> data(in);
  kernel_fma(data.data(), data.size(), config.intensity,
             static_cast<int>(config.nsteps - counter));

  // received vector in
  if (--counter == 0) {
    size_t result = done_counter.fetch_add(1, std::memory_order_relaxed);
    //        fprintf(stderr, "%d: done counter %lu/%lu\n",
    //        hpx::get_locality_id(), result + 1, nchains_per_rank);
    if (result + 1 == nchains_per_rank) {
      done_counter = 0;
      if (config.is_single_source) {
        //                fprintf(stderr, "%d: send one to root\n",
        //                hpx::get_locality_id());
        hpx::post<on_done_action>(hpx::find_root_locality());
      } else {
        //                fprintf(stderr, "%d: send one to here\n",
        //                hpx::get_locality_id());
        hpx::post<on_done_action>(here);
      }
    }
    return;
  }
  // send it to remote locality (to)
  hpx::post<on_recv_action>(peer, std::move(data), counter);
}

hpx::counting_semaphore_var<> semaphore;

void on_done()
{
  //    fprintf(stderr, "%d: on_done\n", hpx::get_locality_id());
  semaphore.signal();
}

hpx::counting_semaphore_var<> barrier_sem;

void trigger_barrier() { barrier_sem.signal(); }
HPX_PLAIN_ACTION(trigger_barrier, trigger_barrier_action)

void barrier()
{
  //    fprintf(stderr, "%d: before barrier\n", hpx::get_locality_id());
  auto peers = hpx::find_remote_localities();
  std::vector<hpx::future<void>> futs;
  for (auto l : peers) {
    futs.emplace_back(hpx::async<trigger_barrier_action>(l));
  }
  hpx::wait_all(futs);
  if (!peers.empty()) barrier_sem.wait(peers.size());
  //    fprintf(stderr, "%d: after barrier\n", hpx::get_locality_id());
}

void run_bench()
{
  auto localities = hpx::find_all_localities();
  auto rank = hpx::get_locality_id();
  double inject_time = 0;
  double time = 0;
  for (size_t j = 0; j < config.nwarmups + config.niters; ++j) {
    if (!config.is_single_source) barrier();
    //            hpx::distributed::barrier::synchronize();
    hpx::chrono::high_resolution_timer timer_total;
    for (size_t i = 0; i < nchains_per_rank; i += config.batch_size) {
      while (config.inject_rate > 0 &&
             static_cast<double>(i) / timer_total.elapsed() >
                 static_cast<double>(config.inject_rate)) {
        continue;
      }
      size_t nchains_left = nchains_per_rank - i;
      hpx::post<on_inject_action>(here, ((nchains_left) > config.batch_size)
                                            ? config.batch_size
                                            : nchains_left);
    }

    if (j >= config.nwarmups) {
      if (!config.is_single_source) barrier();
      //                hpx::distributed::barrier::synchronize();
      inject_time += timer_total.elapsed();
    }

    semaphore.wait();
    //        fprintf(stderr, "%d: wait done\n", hpx::get_locality_id());
    if (j >= config.nwarmups) {
      if (!config.is_single_source) barrier();
      //                hpx::distributed::barrier::synchronize();
      time += timer_total.elapsed();
    }
  }

  if (rank == 0) {
    double achieved_inject_rate =
        static_cast<double>(config.nchains * config.niters) / inject_time / 1e3;
    double latency =
        time * 1e6 / static_cast<double>(config.nsteps * config.niters);
    double msg_rate;
    if (config.is_single_source) {
      msg_rate = static_cast<double>(((config.nsteps + 1) / 2) *
                                     config.nchains * config.niters) /
                 time / 1e3;
    } else {
      msg_rate =
          static_cast<double>(config.nsteps * ((config.nchains + 1) / 2) *
                              config.niters) /
          time / 1e3;
    }
    double bandwidth = static_cast<double>(config.nbytes) * msg_rate / 1e3;
    double total_core_time =
        time * localities.size() * hpx::get_os_thread_count();
    double total_task_time = config.task_comp_time_us * config.niters *
                             config.nchains * config.nsteps / 1e6;
    double comp_efficiency = total_task_time / total_core_time;
    if (config.verbose) {
      std::cout << "[hpx_pingpong]" << std::endl
                << "localities=" << localities.size() << std::endl
                << "total_time(s)=" << time << std::endl
                << "nwarmups=" << config.nwarmups << std::endl
                << "niters=" << config.niters << std::endl
                << "nbytes=" << config.nbytes << std::endl
                << "nchains=" << config.nchains << std::endl
                << "nsteps=" << config.nsteps << std::endl
                << "is_single_source=" << config.is_single_source << std::endl;
      if (config.intensity) {
        std::cout << "intensity=" << config.intensity << std::endl
                  << "comp_time_per_task(us)=" << config.task_comp_time_us
                  << std::endl
                  << "total_task_time(s)=" << total_task_time << std::endl
                  << "total_core_time(s)=" << total_core_time << std::endl
                  << "comp_efficiency=" << comp_efficiency << std::endl;
      }
      std::cout << "latency(us)=" << latency << std::endl
                << "inject_rate(K/s)=" << achieved_inject_rate << std::endl
                << "msg_rate(K/s)=" << msg_rate << std::endl
                << "bandwidth(MB/s)=" << bandwidth << std::endl;
    } else {
      std::cout << "[hpx_pingpong]"
                << ":localities=" << localities.size()
                << ":total_time(secs)=" << time << ":nbytes=" << config.nbytes
                << ":nwarmups=" << config.nwarmups
                << ":niters=" << config.niters << ":nchains=" << config.nchains
                << ":nsteps=" << config.nsteps
                << ":is_single_source=" << config.is_single_source;
      if (config.intensity) {
        std::cout << ":intensity=" << config.intensity
                  << ":comp_time_per_task(us)=" << config.task_comp_time_us
                  << ":total_task_time(s)=" << total_task_time
                  << ":total_core_time(s)=" << total_core_time
                  << ":comp_efficiency=" << comp_efficiency;
      }
      std::cout << ":latency(us)=" << latency
                << ":inject_rate(K/s)=" << achieved_inject_rate
                << ":msg_rate(M/s)=" << msg_rate
                << ":bandwidth(MB/s)=" << bandwidth << std::endl;
    }
  }
  //    fprintf(stderr, "%d: after run_bench\n", hpx::get_locality_id());
  if (!config.is_single_source) barrier();
  //    fprintf(stderr, "%d: exit\n", hpx::get_locality_id());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& b_arg)
{
  config.nbytes = b_arg["nbytes"].as<std::size_t>();
  config.nsteps = b_arg["nsteps"].as<std::size_t>();
  config.nchains = b_arg["nchains"].as<std::size_t>();
  config.inject_rate = b_arg["inject-rate"].as<std::size_t>();
  config.batch_size = b_arg["batch-size"].as<std::size_t>();
  config.nwarmups = b_arg["nwarmups"].as<std::size_t>();
  config.niters = b_arg["niters"].as<std::size_t>();
  config.intensity = b_arg["intensity"].as<std::size_t>();
  config.task_comp_time_us = b_arg["task-comp-time"].as<std::size_t>();
  config.intensity_eval_time_s = b_arg["intensity-eval-time"].as<std::size_t>();
  config.is_single_source = b_arg["is-single-source"].as<bool>();
  config.verbose = b_arg["verbose"].as<bool>();

  evaluate_kernal();

  if (config.nsteps == 0) {
    std::cout << "nsteps is 0!" << std::endl;
    hpx::finalize();
    return 0;
  }

  if (config.nchains == 0) {
    std::cout << "nchains is 0!" << std::endl;
    hpx::finalize();
    return 0;
  }

  if (hpx::get_num_localities().get() > 2) {
    std::cout << "Too many localities!" << std::endl;
    hpx::finalize();
    return 0;
  }

  if (config.nchains == 1) config.is_single_source = true;

  for (auto l : hpx::find_all_localities()) {
    set_config_action act;
    act(l, config);
  }

  if (!config.is_single_source) {
    std::vector<hpx::future<void>> futs;
    for (auto l : hpx::find_all_localities()) {
      futs.emplace_back(hpx::async<run_bench_action>(l));
    }
    hpx::wait_all(futs);
  } else {
    run_bench();
  }

  hpx::finalize();
  return 0;
}

int main(int argc, char* argv[])
{
  namespace po = hpx::program_options;
  po::options_description description("HPX pingpong example");

  description.add_options()(
      "nbytes", po::value<std::size_t>()->default_value(nbytes_default),
      "number of elements (doubles) to send/receive (integer)")(
      "nsteps", po::value<std::size_t>()->default_value(nsteps_default),
      "number of ping-pong iterations")(
      "nchains", po::value<std::size_t>()->default_value(nchains_default),
      "nchains size of ping-pong")(
      "inject-rate",
      po::value<std::size_t>()->default_value(inject_rate_default),
      "the rate of injecting the first message of ping-pong")(
      "batch-size", po::value<std::size_t>()->default_value(batch_size_default),
      "the number of messages to inject per inject thread")(
      "nwarmups", po::value<std::size_t>()->default_value(nwarmups_default),
      "the iteration count of warmup runs")(
      "niters", po::value<std::size_t>()->default_value(niters_default),
      "the iteration count of measurement iterations.")(
      "intensity", po::value<std::size_t>()->default_value(intensity_default),
      "the computation intensity.")(
      "task-comp-time",
      po::value<std::size_t>()->default_value(task_comp_time_us_default),
      "the computation time per task (us).")(
      "intensity-eval-time",
      po::value<std::size_t>()->default_value(intensity_eval_time_s_default),
      "time to evaluate the intensity/comp_time relation (s).")(
      "is-single-source",
      po::value<bool>()->default_value(is_single_source_default),
      "Spawn all message chains from a single source or not")(
      "verbose", po::value<bool>()->default_value(verbose_default),
      "verbosity of output,if false output is for awk");

  hpx::init_params init_args;
  init_args.desc_cmdline = description;

  return hpx::init(argc, argv, init_args);
}

#endif
