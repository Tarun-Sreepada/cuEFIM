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

struct params
{
    // inputfile, output file, minUtil, seperator
    std::string input_file;
    std::string output_file;
    std::string separator;

    char seperatorChar;

    uint64_t min_utility = 0;


    std::string method;

};

struct results
{
    std::vector<
        std::pair<
            std::vector<std::string>, uint64_t>
                > frequentItemsets;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
};

// Hash function for vectors
struct VectorHash {
    size_t operator()(const std::vector<uint32_t> &v) const {
        std::hash<uint32_t> hasher;
        size_t seed = 0;
        for (uint32_t i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
