#pragma once

#include "config.cuh"
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


// Function declarations

// Function to determine the separator character
char get_separator(const std::string &separator);

// Function to read the file and calculate TWU
std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> read_file(
    const params &p,
    std::unordered_map<std::string, uint64_t> &twu,
    char separator_char
);

// Function to filter and sort TWU
std::vector<std::pair<std::string, uint64_t>> filter_and_sort_twu(
    const std::unordered_map<std::string, uint64_t> &twu,
    uint64_t min_utility
);

// Function to process transactions
std::tuple<
    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>,
    std::unordered_map<uint64_t, uint64_t>,
    std::unordered_map<uint64_t, uint64_t>
> process_transactions(
    const std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> &file_data,
    const std::unordered_map<std::string, uint32_t> &strToInt,
    uint64_t min_utility);

// Main parsing function
std::tuple<
    std::unordered_map<std::string, uint32_t>,
    std::unordered_map<uint32_t, std::string>,
    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>,
    std::vector<std::pair<std::string, uint64_t>>,
    std::unordered_map<uint64_t, uint64_t>,
    std::unordered_map<uint64_t, uint64_t>
> parse_file(const params &p);
