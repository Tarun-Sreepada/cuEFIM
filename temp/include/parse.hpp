#pragma once
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring> // For strerror
#include <cstdlib> // For exit
#include <getopt.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <stdexcept>


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

// parse file , return std::vector<item_value>
std::tuple<
    std::unordered_map<std::string, uint32_t>, 
    std::unordered_map<uint32_t, std::string>, 
    std::unordered_map<std::vector<uint32_t>, 
    std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>, 
    std::vector<std::pair<std::string, uint64_t>>, 
    std::vector<uint64_t>, std::vector<uint64_t>
    > 
parse_file(const params &p);




char get_separator(const std::string &separator);
std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> read_file(const params &p, std::unordered_map<std::string, uint64_t> &twu, char separator_char);
std::vector<std::pair<std::string, uint64_t>> filter_and_sort_twu(const std::unordered_map<std::string, uint64_t> &twu, uint64_t min_utility);
std::tuple<std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>, std::vector<uint64_t>, std::vector<uint64_t>>
 process_transactions(
    const std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> &file_data,
    const std::unordered_map<std::string, uint32_t> &strToInt, uint64_t min_utility);
void print_twu(const std::vector<std::pair<std::string, uint64_t>> &sorted_twu);
void print_transactions(const std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash> &filtered_transactions);
