#pragma once
#include "../main.cuh"
#include "args.cuh"
#include "file.hpp"

struct parsed_file {
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> key_value_pairs;
    std::unordered_map<uint32_t, uint32_t> twu;
};


parsed_file parse_file_cpu(raw_file &file, results &r, Config::Params &p);