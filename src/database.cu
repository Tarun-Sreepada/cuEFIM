#include "../main.cuh"

__device__ uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash function
__device__ uint32_t hashFunction(uint32_t key, uint32_t tableSize) {
    return pcg_hash(key) % tableSize;
}


// // Device function to query an item in the hash table
// __device__ int64_t query_item(key_value *item_index, uint32_t start_search, uint32_t end_search, uint32_t item) {

//     uint32_t tableSize = end_search - start_search;

//     uint32_t hashIdx = hashFunction(item, tableSize);

//     while (true) {
//         if (item_index[hashIdx + start_search].key == 0) {
//             return -1; // Item not found
//         }
//         if (item_index[hashIdx + start_search].key == item) {
//             return item_index[hashIdx + start_search].value;
//         }
//         // Handle collisions (linear probing)
//         hashIdx = (hashIdx + 1) % tableSize;
//     }
// }


// __global__ void print_db(database *d_db) {
//     database db = *d_db;
//     for (int i = 0; i < db.transactions_count; i++) {
//         for (int j = db.transaction_start[i]; j < db.transaction_end[i]; j++) {
//             printf("%u:%u ", db.item_utility[j].key, db.item_utility[j].value);
//         }
//         printf("\n");
//     }

// }

// // Kernel to insert transactions into the hash table
// __global__ void hash_transactions(database *d_db) {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= d_db->transactions_count) {
//         return;
//     }

//     uint32_t bucket_size = (d_db->transaction_end[tid] - d_db->transaction_start[tid]) * bucket_factor;
//     uint32_t item_index_insert_start = d_db->transaction_start[tid] * bucket_factor;

//     for (int i = d_db->transaction_start[tid]; i < d_db->transaction_end[tid]; i++) {
//         uint32_t item = d_db->item_utility[i].key;
//         uint32_t hashIdx = hashFunction(item, bucket_size);


//         while (true) {
//             if (d_db->item_index[hashIdx + item_index_insert_start].key == 0) {
//                 d_db->item_index[hashIdx + item_index_insert_start].key = item;
//                 d_db->item_index[hashIdx + item_index_insert_start].value = i;
//                 break;
//             }
//             // Handle collisions (linear probing)
//             hashIdx = (hashIdx + 1) % (bucket_size);
//         }
//     }

// }   

// __global__ void print_db_full(database *d_db) {
//     database db = *d_db;
//     for (int i = 0; i < db.transactions_count; i++) {

//         for (int j = db.transaction_start[i]; j < db.transaction_end[i]; j++) {
//             printf("%u:%u ", db.item_utility[j].key, db.item_utility[j].value);
//         }

//         printf("\n");

//         for (int j = db.transaction_start[i] * bucket_factor; j < db.transaction_end[i] * bucket_factor; j++) {
//             if (db.item_index[j].key == 0) {
//                 continue;
//             }
//             printf("%u|%u:%u ", j - db.transaction_start[i] * bucket_factor, db.item_index[j].key, db.item_index[j].value - db.transaction_start[i]);
//         }

//         printf("\n\n");
//     }

// }
