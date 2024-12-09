#pragma once
#include "../main.cuh"
#include "args.cuh"
#include "memory.cuh"
#include "build.hpp"


struct gpu_db
{

    CudaMemory<key_value> compressed_spare_row_db;
    CudaMemory<key_value> transaction_hash_db;

    CudaMemory<size_t> csr_transaction_start;
    CudaMemory<size_t> csr_transaction_end;
    CudaMemory<bool> transaction_hits;
    
    size_t load_factor;
    size_t transaction_count;
    size_t total_items;
    size_t max_transaction_size;


    gpu_db()
    {
        load_factor = 2;
        transaction_count = 0;
        max_transaction_size = 0;
        total_items = 0;
    }

    ~gpu_db()
    {
    }

};

struct workload
{
    CudaMemory<uint32_t> primary;
    CudaMemory<uint32_t> secondary;

    CudaMemory<uint32_t> primary_utility;
    CudaMemory<uint32_t> subtree_utility;
    CudaMemory<uint32_t> local_utility;

    size_t number_of_primaries;
    size_t primary_size;

    size_t number_of_secondaries;    

};


void mine(build_file &bf, results &r, Config::Params &p);
