#pragma once
#include "../main.cuh"
#include "args.cuh"
#include "parse.hpp"

struct build_file
{

    std::map<uint32_t, uint32_t> ordered_twu;
    std::vector<std::pair<uint32_t, uint32_t>> ordered_twu_vector;

    std::unordered_map<uint32_t, uint32_t> item_to_itemID;
    std::unordered_map<uint32_t, uint32_t> itemID_to_item;

    std::map<uint32_t, uint32_t> subtree_utility;

    std::vector<key_value> compressed_spare_row_db;
    std::vector<size_t> csr_transaction_start;
    std::vector<size_t> csr_transaction_end;


    size_t transaction_count;
    size_t total_items;
    size_t max_transaction_size;

    std::vector<uint32_t> primary;
    std::vector<uint32_t> secondary;


    build_file()
    {
        transaction_count = 0;
        max_transaction_size = 0;
        total_items = 0;
    }

    ~build_file()
    {
        ordered_twu.clear();
        ordered_twu_vector.clear();
        item_to_itemID.clear();
        itemID_to_item.clear();
        subtree_utility.clear();

        compressed_spare_row_db.clear();
        csr_transaction_start.clear();
        csr_transaction_end.clear();

    }

};


build_file build_cpu(parsed_file &pf, results &r, Config::Params &p);