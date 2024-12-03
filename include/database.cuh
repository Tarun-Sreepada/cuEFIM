#pragma once
#include "../main.cuh"

// #define bucket_factor 2

// // CUDA error checking macro
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//     if (code != cudaSuccess)
//     {
//         fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }

// struct item_utility
// {
//     std::string item;
//     uint32_t utility;
// };

// struct line_data
// {
//     std::vector<item_utility> transaction;
// };

// struct file_data
// {
//     std::vector<line_data> data;
// };

// struct key_value
// {
//     uint32_t key;
//     uint32_t value;
// };

// struct pattern
// {
//     std::vector<uint32_t> items;
//     std::vector<std::string> items_names;
//     uint32_t utility;
// };


// struct database
// {
//     uint32_t transactions_count;

//     uint32_t *transaction_start;
//     uint32_t *transaction_end;

//     key_value *item_utility;
//     key_value *item_index; // Hash table
// };

// __global__ void hash_transactions(database *d_db);

// __global__ void print_db(database *d_db);
// __global__ void print_db_full(database *d_db);

// __device__ int64_t query_item(key_value *item_index, uint32_t start_search, uint32_t end_search, uint32_t item);