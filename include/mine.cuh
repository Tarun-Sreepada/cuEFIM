#pragma once
#include "config.cuh"

#define block_size 32

// mine_patterns(p, filtered_transactions, primary, secondary, frequent_patterns);
void mine_patterns(params p, std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash>,
                    std::vector<uint32_t> primary, std::vector<uint32_t> secondary,
                    std::vector<pattern> &frequent_patterns, std::unordered_map<uint32_t, std::string> &intToStr);

