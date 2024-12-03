#pragma once
#define KIBIBYTE 1024ULL
#define KILOBYTE 1000ULL

#ifdef __CUDACC__
#include <cuda_runtime.h> // Only include CUDA headers when using nvcc
#endif

#include <iostream> // For cout
#include <string>
#include <vector> 
#include <unordered_map> 
#include <cstdint> // For uint32_t
#include <liburing.h> // For io_uring / file io
#include <stdexcept> // For runtime_error / invalid_argument / out_of_range
#include <chrono> // For high_resolution_clock
#include <algorithm> // For sort
#include <numeric> // For accumulate
#include <tuple> // For tuple
#include <getopt.h> // For options
#include <algorithm> // For sort / min / max
#include <fstream> // For file io
#include <charconv> // For from_chars
#include <iomanip> // For setprecision



enum class gpu_memory_allocation
{
    Device,
    Unified
};


enum class mine_method
{
    no_hash_table,
    no_hash_table_shared_memory,
    hash_table,
    hash_table_shared_memory,
};


enum class file_read_parse_method
{
    CPU,
    GPU
};


struct params
{
    std::string input_file;
    std::string output_file;
    std::string separator; // TODO: set default to /dev/stdout

    char separator_char = ',';

    uint32_t min_utility = 0;

    size_t page_size = 128 * KIBIBYTE;
    size_t queue_depth = 512;

    file_read_parse_method parse_method = file_read_parse_method::CPU;
    gpu_memory_allocation GPU_memory_allocation = gpu_memory_allocation::Device;
    mine_method method = mine_method::hash_table_shared_memory;
};


/*
    @brief Print help message
*/


struct results
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> file_read_time; // SSD -> RAM
    std::chrono::time_point<std::chrono::high_resolution_clock> parse_time; // RAM -> Structure
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

    uint32_t total_patterns;

    uint32_t gpu_memory_consumption_before_starting;
    std::vector<uint32_t> intermediate_gpu_memory_consumption;
    uint32_t total_gpu_memory_consumption;

    uint32_t cpu_memory_consumption;

    uint32_t total_memory_consumption; // CPU + GPU
};

// Hash function for vectors
struct VectorHash
{
    uint32_t operator()(const std::vector<uint32_t> &v) const
    {
        std::hash<uint32_t> hasher;
        uint32_t seed = 0;
        for (uint32_t i : v)
        {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

