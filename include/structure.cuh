#pragma once
#include "../main.cuh"
#include "memory.cuh"

struct key_value
{
    uint32_t key;
    uint32_t value;
};

struct transaction
{
    std::vector<key_value> item_utility;
};

struct hash_table
{
    size_t size;
    key_value *entries;

};

// struct workload {
//     CudaMemory<uint32_t> d_primary;
//     size_t primary_size;
//     size_t primary_count;

//     CudaMemory<uint32_t> d_secondary_ref;

//     CudaMemory<uint32_t> d_secondary;
//     size_t secondary_size;

//     // Default constructor with no allocation
//     workload()
//         : d_primary(),               // No allocation
//           primary_size(0),
//           primary_count(0),
//           d_secondary_ref(),         // No allocation
//           d_secondary(),             // No allocation
//           secondary_size(0) {}

//     // Constructor with allocation
//     workload(size_t primarySize, size_t secondarySize, gpu_memory_allocation memType)
//         : d_primary(primarySize, memType),
//           primary_size(primarySize),
//           primary_count(0),
//           d_secondary_ref(secondarySize, memType),
//           d_secondary(secondarySize, memType),
//           secondary_size(secondarySize) {}
// };

// struct database
// {
//     std::vector<key_value> compressed_spare_row_db;
//     std::vector<size_t> csr_transaction_start;
//     std::vector<size_t> csr_transaction_end;
//     size_t db_size;

//     key_value *d_compressed_spare_row_db;
//     key_value *d_hashed_csr_db;

//     size_t *d_csr_transaction_start;
//     size_t *d_csr_transaction_end;

//     size_t load_factor;


//     database()
//     {
//     }

//     ~database()
//     {
//         compressed_spare_row_db.clear();
//         csr_transaction_start.clear();
//         csr_transaction_end.clear();

//     }
// };
