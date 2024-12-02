#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <cstdlib> // For exit
#include <getopt.h>
#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>
#include <chrono>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "database.cuh"


struct params
{
    std::string input_file;
    std::string output_file;
    std::string separator;

    char seperatorChar;

    uint32_t min_utility = 0;

    std::string method = "GPU";
};

struct results
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

    uint32_t total_patterns;
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
