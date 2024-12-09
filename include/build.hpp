#pragma once
#include "../main.cuh"
#include "args.cuh"
#include "parse.hpp"

struct build_file
{
    std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash> transactions;

    std::map<uint32_t, uint32_t> ordered_twu;
    std::vector<std::pair<uint32_t, uint32_t>> ordered_twu_vector;

    std::unordered_map<uint32_t, uint32_t> item_to_itemID;
    std::unordered_map<uint32_t, uint32_t> itemID_to_item;

    std::map<uint32_t, uint32_t> subtree_utility;

};


build_file build_cpu(parsed_file &pf, results &r, Config::Params &p);