#pragma once
#include "../main.cuh"
#include "memory.cuh"

struct d_database
{

    CudaMemory<key_value> d_compressed_spare_row_db;
    CudaMemory<key_value> d_hashed_csr_db;

    CudaMemory<size_t> d_csr_transaction_start;
    CudaMemory<size_t> d_csr_transaction_end;

    size_t transaction_count;

    size_t largest_transaction_size;

    CudaMemory<bool> valid_transaction;

    CudaMemory<uint32_t> primary;
    size_t primary_size;

    CudaMemory<uint32_t> secondary;
    size_t secondary_size;

    size_t load_factor;

    // Copy to device memory and pass pointer
    d_database* toDevice() {
        d_database* d_db;
        cudaMalloc((void**)&d_db, sizeof(d_database));
        cudaMemcpy(d_db, this, sizeof(d_database), cudaMemcpyHostToDevice);
        return d_db;
    }


};
