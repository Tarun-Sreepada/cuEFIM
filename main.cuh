#pragma once
#define KIBIBYTE 1024ULL
#define KILOBYTE 1000ULL

#include <cuda_runtime.h> // For cudaError_t / cudaMemGetInfo

#include <iostream> // For cout
#include <string>
#include <vector> 
#include <map>
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


struct results
{

    uint32_t get_cuda_memory_usage() {
        size_t free_mem, total_mem;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        
        if (err != cudaSuccess) {
            std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        }

        // Return used memory (total - free) in bytes
        return static_cast<uint32_t>(total_mem - free_mem);
    }

    std::vector<
        std::pair<std::string, 
            std::tuple<
                std::chrono::time_point<std::chrono::high_resolution_clock>, uint32_t, uint32_t>>> memory_usage;

    // Function to record RSS and CUDA memory usage with a custom label
    void record_memory_usage(const std::string& label) {
        auto now = std::chrono::high_resolution_clock::now();
        uint32_t rss = get_rss_memory_usage();
        uint32_t cuda_mem = get_cuda_memory_usage();
        memory_usage.emplace_back(label, std::make_tuple(now, rss, cuda_mem));
    }

private:
    // Helper function to retrieve RSS memory usage
    uint32_t get_rss_memory_usage() {
        std::ifstream stat_file("/proc/self/statm"); // Linux-specific
        if (!stat_file.is_open()) {
            return 0; // Unable to read RSS
        }

        std::string line;
        std::getline(stat_file, line);
        std::istringstream iss(line);

        uint32_t pages;
        iss >> pages; // Skip virtual memory size
        iss >> pages; // Resident Set Size in pages

        stat_file.close();

        // Convert pages to bytes
        return pages * sysconf(_SC_PAGESIZE);
    }

};