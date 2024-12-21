#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <random>
#include <vector>
#include <chrono>
#include "hpcc_util.hpp"

// Constants for the Random Access benchmark
constexpr uint64_t TABLE_SIZE_LOG_DEFAULT = 10;  // Default table size
constexpr uint64_t TABLE_SIZE_DEFAULT =
    1ULL << TABLE_SIZE_LOG_DEFAULT;  // Default table size (16M entries)
constexpr uint64_t BATCH_SIZE_DEFAULT =
    1000;  // Number of max pending operations

// Component for distributed table storage
struct table_component : hpx::components::component_base<table_component> {
  std::vector<uint64_t> local_table;

  table_component(uint64_t local_table_size) : local_table(local_table_size)
  {
    uint64_t global_start = hpx::get_locality_id() * local_table_size;
    for (auto& element : local_table) {
      element = global_start++;
    }
  }

  // Action to update a single value in the local table
  void update_value(uint64_t index, uint64_t value)
  {
    if (index < local_table.size()) {
      local_table[index] ^= value;
    } else {
      std::cerr << "Invalid index " << index << " on locality "
                << hpx::get_locality_id() << std::endl;
    }
  }
  HPX_DEFINE_COMPONENT_ACTION(table_component, update_value);
};

// Register component and action
typedef hpx::components::component<table_component> table_component_type;
HPX_REGISTER_COMPONENT(table_component_type, table_component);
typedef table_component::update_value_action update_value_action;
HPX_REGISTER_ACTION(update_value_action);

// Function to perform updates on distributed table
void perform_updates(std::vector<hpx::id_type> const& table_components,
                     uint64_t start_random, size_t num_local_updates,
                     uint64_t total_size, uint64_t batch_size)
{
  uint64_t random = start_random;
  size_t entries_per_locality = total_size / table_components.size();
  std::vector<hpx::future<void>> futures;
  futures.reserve(batch_size);

  for (size_t i = 0; i < num_local_updates; ++i) {
    random = hpcc::generate_next_random(random);
    uint64_t index = random & (total_size - 1);
    size_t locality_id = index / entries_per_locality;
    uint64_t local_index = index % entries_per_locality;

    if (locality_id < table_components.size()) {
      futures.push_back(hpx::async<update_value_action>(
          table_components[locality_id], local_index, random));
    }

    // Batch futures to avoid overwhelming the system
    if (batch_size > 0 && futures.size() >= batch_size) {
      hpx::wait_all(futures);
      futures.clear();
    }
  }

  // Wait for remaining futures
  hpx::wait_all(futures);
}
HPX_PLAIN_ACTION(perform_updates, perform_updates_action);

int hpx_main(hpx::program_options::variables_map& vm)
{
  uint64_t table_size_log = vm["table-size-log"].as<uint64_t>();
  uint64_t table_size = 1ULL << table_size_log;
  uint64_t nupdates = vm["nupdates"].as<uint64_t>();
  if (nupdates == 0) {
    uint64_t xupdates = vm["xupdates"].as<uint64_t>();
    nupdates = xupdates * table_size;
  }
  uint64_t batch_size = vm["batch-size"].as<uint64_t>();

  // Get information about localities
  std::size_t num_localities = hpx::get_num_localities().get();
  std::size_t locality_id = hpx::get_locality_id();

  // Create table components on all localities
  std::vector<hpx::future<hpx::id_type>> component_futures;
  for (auto locality : hpx::find_all_localities()) {
    component_futures.push_back(
        hpx::new_<table_component>(locality, table_size / num_localities));
  }
  std::vector<hpx::id_type> table_components = hpx::unwrap(component_futures);

  // Calculate work distribution
  size_t updates_per_locality = nupdates / num_localities;

  // Initialize start random
  std::vector<uint64_t> start_randoms;
  for (size_t i = 0; i < num_localities; ++i) {
    start_randoms.push_back(hpcc::starts(updates_per_locality * i));
  }

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Launch the benchmark on all localities
  auto all_localities = hpx::find_all_localities();
  std::vector<hpx::future<void>> update_futures;
  for (size_t i = 0; i < num_localities; ++i) {
    update_futures.push_back(hpx::async<perform_updates_action>(
        all_localities[i], table_components, start_randoms[i],
        updates_per_locality, table_size, batch_size));
  }

  // Wait for all updates to complete
  hpx::wait_all(update_futures);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // Calculate GUPS
  double gups = (nupdates / 1e9) / (duration.count() / 1000.0);

  if (locality_id == 0) {
    std::cout << "HPCC Random Access Benchmark Results:\n"
              << "Number of localities: " << num_localities << "\n"
              << "Table size: " << table_size << " elements\n"
              << "Number of updates: " << nupdates << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Time taken: " << duration.count() << " ms\n"
              << "Performance: " << gups << " GUPS\n";
  }

  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  using hpx::program_options::value;

  // Configure application-specific options
  hpx::program_options::options_description desc_commandline(
      "Usage: " HPX_APPLICATION_STRING " [options]");

  desc_commandline.add_options()(
      "table-size-log",
      value<uint64_t>()->default_value(TABLE_SIZE_LOG_DEFAULT),
      "the log 2 size of the table")(
      "xupdates", value<uint64_t>()->default_value(4),
      "the number of lookups to perform relative to table size")(
      "nupdates", value<uint64_t>()->default_value(0),
      "the number of lookups to perform")(
      "batch-size", value<uint64_t>()->default_value(BATCH_SIZE_DEFAULT),
      "batch size");
  // Initialize and run HPX
  hpx::init_params init_args;
  init_args.desc_cmdline = desc_commandline;

  return hpx::init(argc, argv, init_args);
}