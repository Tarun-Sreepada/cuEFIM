#include "mine.cuh"

__device__ uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash function
__device__ uint32_t hashFunction(uint32_t key, uint32_t tableSize)
{
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


__global__ void print_database(d_database *d_db)
{
    printf("Primary Size: %u\n", d_db->primary_size);
    printf("Secondary Size: %u\n", d_db->secondary_size);
    printf("Largest Transaction Size: %u\n", d_db->largest_transaction_size);
    printf("Load Factor: %u\n", d_db->load_factor);

    printf("Primary: ");
    for (int i = 0; i < d_db->primary_size; i++)
    {
        printf("%ld ", d_db->primary.ptr()[i]);
    }
    printf("\n");

    printf("Secondary: ");
    for (int i = 0; i < d_db->secondary_size; i++)
    {
        printf("%ld ", d_db->secondary.ptr()[i]);
    }

    printf("\n");

    // print transactions
    for (int i = 0; i < d_db->transaction_count; i++)
    {
        for (int j = d_db->d_csr_transaction_start.ptr()[i]; j < d_db->d_csr_transaction_end.ptr()[i]; j++)
        {
            printf("%u:%u ", d_db->d_compressed_spare_row_db.ptr()[j].key, d_db->d_compressed_spare_row_db.ptr()[j].value);
        }
        printf("\n");
    }

    // print hashed transactions
    for (int i = 0; i < d_db->transaction_count; i++)
    {
        for (int j = d_db->d_csr_transaction_start.ptr()[i] * d_db->load_factor; j < d_db->d_csr_transaction_end.ptr()[i] * d_db->load_factor; j++)
        {
            // printf("%u:%u ", d_db->d_compressed_spare_row_db.ptr()[j].key, d_db->d_compressed_spare_row_db.ptr()[j].value);
            printf("%u:%u ", d_db->d_hashed_csr_db.ptr()[j].key, d_db->d_hashed_csr_db.ptr()[j].value);
        }
        printf("\n");
    }
}


// Kernel to insert transactions into the hash table
__global__ void hash_transactions(d_database *d_db)

{

    printf("Primary Size: %u\n", d_db->primary_size);
    printf("Secondary Size: %u\n", d_db->secondary_size);
    printf("Largest Transaction Size: %u\n", d_db->largest_transaction_size);
    printf("Load Factor: %u\n", d_db->load_factor);
    printf("Start: %u\tEnd: %u\n", d_db->d_csr_transaction_start.ptr()[0], d_db->d_csr_transaction_end.ptr()[1]);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_db->transaction_count)
    {
        return;
    }
    printf("TID: %u\n", tid);
    printf("Start: %u\tEnd: %u\n", d_db->d_csr_transaction_start.ptr()[tid], d_db->d_csr_transaction_end.ptr()[tid]);

    uint32_t bucket_size = (d_db->d_csr_transaction_start.ptr()[tid] - d_db->d_csr_transaction_end.ptr()[tid]) * d_db->load_factor;
    uint32_t item_index_insert_start = d_db->d_csr_transaction_start.ptr()[tid] * d_db->load_factor;
    printf("Bucket Size: %u\tLoad Factor: %u\n", bucket_size, d_db->load_factor);
    printf("Item Index Insert Start: %u\n", item_index_insert_start);

    for (int i = d_db->d_csr_transaction_start.ptr()[tid]; i < d_db->d_csr_transaction_end.ptr()[tid]; i++)
    {
        uint32_t item = d_db->d_compressed_spare_row_db.ptr()[i].key;
        uint32_t hashIdx = hashFunction(item, bucket_size);

        // d_db->d_hashed_csr_db.ptr()
        while (true)
        {
            if (d_db->d_hashed_csr_db.ptr()[hashIdx + item_index_insert_start].key == 0)
            {
                d_db->d_hashed_csr_db.ptr()[hashIdx + item_index_insert_start].key = item;
                d_db->d_hashed_csr_db.ptr()[hashIdx + item_index_insert_start].value = i;
                break;
            }
            // Handle collisions (linear probing)
            hashIdx = (hashIdx + 1) % (bucket_size);
        }
    }
}


void mine(CudaMemory<d_database> &db, results &r, Config::Params &p)
{
#ifdef DEBUG
    std::cout << "Mining" << std::endl;
#endif

    // d_database d_db = db; // just to copy the pointers over to the GPU so we can avoid the CPU-GPU communication

   
    dim3 block(1);
    dim3 grid(1);

    print_database<<<grid, block>>>(d_db);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    hash_transactions<<<grid, block>>>(d_db);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    r.record_memory_usage("Hashed DB");
}

// __global__ void searchGPU_shared_mem_k_v(database *d_db, uint32_t *transaction_hits, uint32_t transactions_count,
//                                          uint32_t *candidates, uint32_t number_of_candidates, uint32_t candidate_size,
//                                          uint32_t *secondary, uint32_t secondary_size,
//                                          uint32_t *secondary_reference,
//                                          uint32_t *candidate_utility,
//                                          uint32_t *candidate_subtree_utility,
//                                          uint32_t *candidate_local_utility)
// {
//     uint32_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
//     uint32_t tid = threadIdx.x;
//     if (block_id >= transactions_count || transaction_hits[block_id] == 0)
//     {
//         return;
//     }
//     transaction_hits[block_id] = 0;

//     uint32_t transaction_start = d_db->transaction_start[block_id];
//     uint32_t transaction_end = d_db->transaction_end[block_id];
//     uint32_t transaction_length = transaction_end - transaction_start;

//     for (uint32_t i = tid; i < transaction_length * bucket_factor; i += blockDim.x)
//     {
//         // shared_memory[i] = d_db->item_index[transaction_start * bucket_factor + i];
//         shared_memory[i].key = d_db->item_index[transaction_start * bucket_factor + i].key;
//         shared_memory[i].value = d_db->item_index[transaction_start * bucket_factor + i].value;
//     }

//     __syncthreads();

//     for (uint32_t i = tid; i < number_of_candidates; i += blockDim.x)
//     {
//         uint32_t curr_cand_util = 0;
//         uint32_t curr_cand_hits = 0;
//         int32_t location = -1;

//         for (uint32_t j = 0; j < candidate_size; j++)
//         {
//             uint32_t candidate = candidates[i * candidate_size + j];
//             location = query_item(shared_memory, 0, transaction_length * bucket_factor, candidate);
//             if (location != -1)
//             {
//                 curr_cand_hits++;
//                 curr_cand_util += d_db->item_utility[location].value;
//             }
//         }
//         if (curr_cand_hits != candidate_size)
//         {
//             continue;
//         }

//         transaction_hits[block_id] = 1;
//         atomicAdd(&candidate_utility[i], curr_cand_util);

//         // calculate the TWU
//         uint32_t ref = secondary_reference[i];
//         uint32_t secondary_index_start = secondary_size * ref;

//         // collect all utilities
//         for (uint32_t j = location + 1; j < transaction_end; j++)
//         {
//             uint32_t item = d_db->item_utility[j].key;
//             if (secondary[secondary_index_start + item]) // if the item is valid secondary
//             {
//                 curr_cand_util += d_db->item_utility[j].value;
//             }
//         }

//         uint32_t temp = 0;

//         uint32_t subtree_local_insert_location = i * secondary_size;

//         for (uint32_t j = location + 1; j < transaction_end; j++)
//         {
//             uint32_t item = d_db->item_utility[j].key;
//             if (secondary[secondary_index_start + item]) // if the item is valid secondary
//             {
//                 atomicAdd(&candidate_local_utility[subtree_local_insert_location + item], curr_cand_util);
//                 atomicAdd(&candidate_subtree_utility[subtree_local_insert_location + item], curr_cand_util - temp);
//                 temp += d_db->item_utility[j].value;
//             }
//         }
//     }
// }

// void mine(database &db, std::unordered_map<uint32_t, uint32_t> &subtree_utility,
//             std::unordered_map<uint32_t, uint32_t> &rank, params &p, results &r)
// {
//     workload w;

//     std::vector<uint32_t> primary;
//     for (auto &item : subtree_utility)
//     {
//         if (item.second >= p.min_utility)
//         {
//             primary.push_back(item.first);
//         }
//     }

//     w.primary_size = 1;
//     w.primary_count = primary.size();
//     w.secondary_size = rank.size() + 1;

//     // allocate
//     w.d_primary = CudaMemory<uint32_t>(primary.size(), p.GPU_memory_allocation);
//     w.d_secondary_ref = CudaMemory<uint32_t>(primary.size(), p.GPU_memory_allocation);
//     w.d_secondary = CudaMemory<uint32_t>(w.secondary_size, p.GPU_memory_allocation);

//     // copy primary and memset secondary to 1
//     cudaMemcpy(w.d_primary.ptr(), primary.data(), primary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
//     cudaMemset(w.d_secondary.ptr(), 1, w.secondary_size * sizeof(uint32_t));
//     cudaMemset(w.d_secondary_ref.ptr(), 0, primary.size() * sizeof(uint32_t));

// }

// __global__ void clean_subtree_local_utility(uint32_t number_of_candidates, uint32_t *number_of_new_candidates_per_candidate,
//                                             uint32_t *subtree_utility, uint32_t *local_utility, uint32_t secondary_size, uint32_t minimum_utility)
// {
//     uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= number_of_candidates) return;

//     for (uint32_t i = tid * secondary_size; i < (tid + 1) * secondary_size; i++)
//     {
//         uint32_t item_value = i - tid * secondary_size;

//         if (subtree_utility[i] >= minimum_utility)
//         {
//             subtree_utility[i] = item_value;
//             number_of_new_candidates_per_candidate[tid + 1]++;
//         }
//         else
//         {
//             subtree_utility[i] = 0;
//         }
//         if (local_utility[i] >= minimum_utility)
//         {
//             local_utility[i] = item_value;
//         }
//         else
//         {
//             local_utility[i] = 0;
//         }
//     }
//     return;
// }

// __global__ void create_new_candidates(uint32_t *candidates, uint32_t *candidate_subtree_utility, uint32_t number_of_candidates,
//                                       uint32_t *new_candidates, uint32_t *new_secondary_reference, uint32_t secondary_size, uint32_t candidate_size,
//                                       uint32_t *number_of_new_candidates_per_candidate)
// {
//     uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= number_of_candidates)
//     {
//         return;
//     }

//     // if no new candidates
//     if (number_of_new_candidates_per_candidate[tid] == number_of_new_candidates_per_candidate[tid + 1])
//     {
//         return;
//     }

//     uint32_t counter = candidate_size * number_of_new_candidates_per_candidate[tid];
//     uint32_t refStart = number_of_new_candidates_per_candidate[tid];

//     for (uint32_t i = tid * secondary_size; i < (tid + 1) * secondary_size; i++)
//     {
//         if (candidate_subtree_utility[i])
//         {
//             for (uint32_t j = tid * (candidate_size - 1); j < (tid + 1) * (candidate_size - 1); j++)
//             {
//                 new_candidates[counter] = candidates[j];
//                 counter++;
//             }
//             new_candidates[counter] = candidate_subtree_utility[i];
//             counter++;
//             new_secondary_reference[refStart] = tid;
//             refStart++;
//         }
//     }

//     return;
// }

// void print_used_gpu_memory() {
//     size_t free_mem, total_mem;
//     cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

//     if (err != cudaSuccess) {
//         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
//         return;
//     }

//     std::cout << "Used GPU memory: " << (total_mem - free_mem) / 1024.0 / 1024.0 << " MB" << std::endl;
// }

// // Define necessary structures and functions (assuming they are defined elsewhere)
// // For example: params, pattern, key_value, database, gpuErrchk, hash_transactions,
// // searchGPU_shared_mem_k_v, clean_subtree_local_utility, create_new_candidates, etc.

// // Function to prepare transactions
// void prepare_transactions(const std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash> &filtered_transactions,
//                           std::vector<uint32_t> &primary,
//                           std::vector<uint32_t> &secondary,
//                           std::vector<uint32_t> &transaction_start,
//                           std::vector<uint32_t> &transaction_end,
//                           std::vector<key_value> &item_utility,
//                           uint32_t &max_transaction_length)
// {
//     std::cout << "Number of transactions: " << filtered_transactions.size() << std::endl;

//     secondary.push_back(1); // Add 1 to the secondary list
//     std::sort(secondary.begin(), secondary.end());
//     std::sort(primary.begin(), primary.end());

//     max_transaction_length = 0;

//     for (const auto &transaction : filtered_transactions)
//     {
//         transaction_start.push_back(static_cast<uint32_t>(item_utility.size()));

//         const std::vector<uint32_t> &items = transaction.first;
//         const std::vector<uint32_t> &utilities = transaction.second;

//         for (size_t i = 0; i < items.size(); ++i)
//         {
//             item_utility.push_back({items[i], utilities[i]});
//         }

//         transaction_end.push_back(static_cast<uint32_t>(item_utility.size()));

//         max_transaction_length = std::max(max_transaction_length, static_cast<uint32_t>(items.size()));
//     }
// }

// // Function to compute shared memory requirement
// size_t compute_shared_memory_requirement(uint32_t max_transaction_length)
// {
//     return max_transaction_length * sizeof(key_value) * bucket_factor;
// }

// // Function to check shared memory requirement against device capability
// void check_shared_memory_requirement(size_t shared_memory_requirement)
// {
//     int device;
//     cudaDeviceProp props;
//     cudaGetDevice(&device);
//     cudaGetDeviceProperties(&props, device);

//     std::cout << "Shared memory requirement: " << shared_memory_requirement << " bytes" << std::endl;
//     std::cout << "Max shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;

//     if (shared_memory_requirement > props.sharedMemPerBlock)
//     {
//         std::cerr << "Shared memory requirement exceeds the maximum shared memory per block" << std::endl;
//     }
// }

// // Function to copy transactions to GPU and calculate memory usage
// size_t copy_transactions_to_gpu(const std::vector<uint32_t> &transaction_start,
//                                 const std::vector<uint32_t> &transaction_end,
//                                 const std::vector<key_value> &item_utility,
//                                 database *&d_db)
// {
//     size_t total_gpu_memory = 0;

//     // Allocate database on GPU
//     gpuErrchk(cudaMallocManaged(&d_db, sizeof(database)));
//     total_gpu_memory += sizeof(database);

//     // Allocate and copy transaction start indices
//     uint32_t *d_transaction_start = nullptr;
//     gpuErrchk(cudaMallocManaged(&d_transaction_start, transaction_start.size() * sizeof(uint32_t)));
//     gpuErrchk(cudaMemcpy(d_transaction_start, transaction_start.data(), transaction_start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
//     total_gpu_memory += transaction_start.size() * sizeof(uint32_t);

//     // Allocate and copy transaction end indices
//     uint32_t *d_transaction_end = nullptr;
//     gpuErrchk(cudaMallocManaged(&d_transaction_end, transaction_end.size() * sizeof(uint32_t)));
//     gpuErrchk(cudaMemcpy(d_transaction_end, transaction_end.data(), transaction_end.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
//     total_gpu_memory += transaction_end.size() * sizeof(uint32_t);

//     // Allocate and copy item utilities
//     key_value *d_item_utility = nullptr;
//     gpuErrchk(cudaMallocManaged(&d_item_utility, item_utility.size() * sizeof(key_value)));
//     gpuErrchk(cudaMemcpy(d_item_utility, item_utility.data(), item_utility.size() * sizeof(key_value), cudaMemcpyHostToDevice));
//     total_gpu_memory += item_utility.size() * sizeof(key_value);

//     // Allocate item index (size depends on bucket_factor)
//     key_value *d_item_index = nullptr;
//     gpuErrchk(cudaMallocManaged(&d_item_index, item_utility.size() * bucket_factor * sizeof(key_value)));
//     gpuErrchk(cudaMemset(d_item_index, 0, item_utility.size() * bucket_factor * sizeof(key_value)));
//     total_gpu_memory += item_utility.size() * bucket_factor * sizeof(key_value);

//     // Set up the database structure on device
//     uint32_t transactions_count = static_cast<uint32_t>(transaction_start.size());

//     gpuErrchk(cudaMemcpy(&(d_db->transactions_count), &transactions_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(&(d_db->transaction_start), &d_transaction_start, sizeof(uint32_t *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(&(d_db->transaction_end), &d_transaction_end, sizeof(uint32_t *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(&(d_db->item_utility), &d_item_utility, sizeof(key_value *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(&(d_db->item_index), &d_item_index, sizeof(key_value *), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaDeviceSynchronize());
//     gpuErrchk(cudaPeekAtLastError());

//     return total_gpu_memory;
// }

// // Function to print GPU memory usage breakdown
// void print_gpu_memory_usage(const std::vector<uint32_t> &transaction_start,
//                             const std::vector<uint32_t> &transaction_end,
//                             const std::vector<key_value> &item_utility,
//                             size_t total_gpu_memory)
// {
//     std::cout << "GPU Memory Usage Breakdown:" << std::endl;
//     std::cout << "  Database struct: " << sizeof(database) << " bytes" << std::endl;
//     std::cout << "  Transaction start indices: " << transaction_start.size() * sizeof(uint32_t) << " bytes" << std::endl;
//     std::cout << "  Transaction end indices: " << transaction_end.size() * sizeof(uint32_t) << " bytes" << std::endl;
//     std::cout << "  Item utilities: " << item_utility.size() * sizeof(key_value) << " bytes" << std::endl;
//     std::cout << "  Item index: " << item_utility.size() * bucket_factor * sizeof(key_value) << " bytes" << std::endl;
//     std::cout << "  Total GPU memory used: " << total_gpu_memory << " bytes" << std::endl;
// }

// // Function to perform mining on GPU
// void mine_gpu_patterns(const params &p,
//                        database *d_db,
//                        const std::vector<uint32_t> &primary,
//                        const std::vector<uint32_t> &secondary,
//                        std::vector<pattern> &frequent_patterns,
//                        const std::unordered_map<uint32_t, std::string> &intToStr,
//                        size_t shared_memory_requirement)
// {
//     uint32_t transactions_count = 0;
//     gpuErrchk(cudaMemcpy(&transactions_count, &(d_db->transactions_count), sizeof(uint32_t), cudaMemcpyDeviceToHost));

//     dim3 block(block_size);
//     dim3 grid((transactions_count + block.x - 1) / block.x);

//     // Hash transactions
//     hash_transactions<<<grid, block>>>(d_db);
//     gpuErrchk(cudaDeviceSynchronize());
//     gpuErrchk(cudaPeekAtLastError());

//     // Initialize candidate patterns
//     thrust::universal_vector<uint32_t> d_candidates = primary;
//     thrust::universal_vector<uint32_t> d_secondary_reference(primary.size(), 0);
//     thrust::universal_vector<uint32_t> d_secondary = secondary;

//     uint32_t number_of_candidates = static_cast<uint32_t>(primary.size());
//     uint32_t candidate_size = 1;

//     thrust::universal_vector<uint32_t> transaction_hits(transactions_count, 1);

//     std::vector<std::pair<thrust::host_vector<uint32_t>, thrust::host_vector<uint32_t>>> original_patterns;

//     // Main mining loop
//     while (number_of_candidates > 0)
//     {
//         std::cout << "Number of candidates: " << number_of_candidates << std::endl;

//         thrust::universal_vector<uint32_t> d_candidate_utility(number_of_candidates, 0);
//         thrust::universal_vector<uint32_t> d_candidate_subtree_utility(number_of_candidates * secondary.size(), 0);
//         thrust::universal_vector<uint32_t> d_candidate_local_utility(number_of_candidates * secondary.size(), 0);

//         // Call the search kernel
//         grid = dim3(transactions_count);
//         searchGPU_shared_mem_k_v<<<grid, block, shared_memory_requirement>>>(d_db,
//                                                                              thrust::raw_pointer_cast(transaction_hits.data()),
//                                                                              transactions_count,
//                                                                              thrust::raw_pointer_cast(d_candidates.data()),
//                                                                              number_of_candidates,
//                                                                              candidate_size,
//                                                                              thrust::raw_pointer_cast(d_secondary.data()),
//                                                                              static_cast<uint32_t>(secondary.size()),
//                                                                              thrust::raw_pointer_cast(d_secondary_reference.data()),
//                                                                              thrust::raw_pointer_cast(d_candidate_utility.data()),
//                                                                              thrust::raw_pointer_cast(d_candidate_subtree_utility.data()),
//                                                                              thrust::raw_pointer_cast(d_candidate_local_utility.data()));

//         gpuErrchk(cudaDeviceSynchronize());
//         gpuErrchk(cudaPeekAtLastError());

//         // Collect candidate utilities
//         thrust::host_vector<uint32_t> h_candidates = d_candidates;
//         thrust::host_vector<uint32_t> h_candidate_utility = d_candidate_utility;
//         original_patterns.emplace_back(h_candidates, h_candidate_utility);

//         candidate_size += 1;

//         // Clean up candidate utilities and prepare for next iteration
//         thrust::universal_vector<uint32_t> d_number_of_new_candidates_per_candidate(number_of_candidates + 1, 0);

//         grid = dim3((number_of_candidates + block_size - 1) / block_size);
//         clean_subtree_local_utility<<<grid, block>>>(number_of_candidates,
//                                                      thrust::raw_pointer_cast(d_number_of_new_candidates_per_candidate.data()),
//                                                      thrust::raw_pointer_cast(d_candidate_subtree_utility.data()),
//                                                      thrust::raw_pointer_cast(d_candidate_local_utility.data()),
//                                                      static_cast<uint32_t>(secondary.size()),
//                                                      p.min_utility);

//         uint32_t number_of_new_candidates = thrust::reduce(d_number_of_new_candidates_per_candidate.begin(),
//                                                            d_number_of_new_candidates_per_candidate.end());
//         thrust::inclusive_scan(d_number_of_new_candidates_per_candidate.begin(),
//                                d_number_of_new_candidates_per_candidate.end(),
//                                d_number_of_new_candidates_per_candidate.begin());

//         if (number_of_new_candidates == 0)
//         {
//             break;
//         }

//         thrust::universal_vector<uint32_t> d_new_candidates(number_of_new_candidates * candidate_size, 0);
//         thrust::universal_vector<uint32_t> d_new_secondary_reference(number_of_new_candidates, 0);

//         create_new_candidates<<<grid, block>>>(thrust::raw_pointer_cast(d_candidates.data()),
//                                                thrust::raw_pointer_cast(d_candidate_subtree_utility.data()),
//                                                number_of_candidates,
//                                                thrust::raw_pointer_cast(d_new_candidates.data()),
//                                                thrust::raw_pointer_cast(d_new_secondary_reference.data()),
//                                                static_cast<uint32_t>(secondary.size()),
//                                                candidate_size,
//                                                thrust::raw_pointer_cast(d_number_of_new_candidates_per_candidate.data()));

//         gpuErrchk(cudaDeviceSynchronize());
//         gpuErrchk(cudaPeekAtLastError());

//         print_used_gpu_memory();

//         d_candidates.swap(d_new_candidates);
//         d_secondary.swap(d_candidate_local_utility);
//         d_secondary_reference.swap(d_new_secondary_reference);
//         number_of_candidates = number_of_new_candidates;
//     }

//     // Collect frequent patterns
//     for (size_t i = 0; i < original_patterns.size(); ++i)
//     {
//         const thrust::host_vector<uint32_t> &h_candidates = original_patterns[i].first;
//         const thrust::host_vector<uint32_t> &h_candidate_utility = original_patterns[i].second;

//         uint32_t size = static_cast<uint32_t>(i + 1);

//         for (size_t j = 0; j < h_candidate_utility.size(); ++j)
//         {
//             if (h_candidate_utility[j] < p.min_utility)
//             {
//                 continue;
//             }

//             pattern pat;
//             for (size_t k = 0; k < size; ++k)
//             {
//                 uint32_t item = h_candidates[j * size + k];
//                 pat.items_names.push_back(intToStr.at(item));
//             }
//             pat.utility = h_candidate_utility[j];
//             frequent_patterns.push_back(pat);
//         }
//     }
// }

// // Function to free GPU memory
// void free_gpu_memory(database *d_db)
// {
//     // Get pointers from device database struct
//     uint32_t *d_transaction_start = nullptr;
//     uint32_t *d_transaction_end = nullptr;
//     key_value *d_item_utility = nullptr;
//     key_value *d_item_index = nullptr;

//     gpuErrchk(cudaMemcpy(&d_transaction_start, &(d_db->transaction_start), sizeof(uint32_t *), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaMemcpy(&d_transaction_end, &(d_db->transaction_end), sizeof(uint32_t *), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaMemcpy(&d_item_utility, &(d_db->item_utility), sizeof(key_value *), cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaMemcpy(&d_item_index, &(d_db->item_index), sizeof(key_value *), cudaMemcpyDeviceToHost));

//     // Free device memory
//     gpuErrchk(cudaFree(d_transaction_start));
//     gpuErrchk(cudaFree(d_transaction_end));
//     gpuErrchk(cudaFree(d_item_utility));
//     gpuErrchk(cudaFree(d_item_index));
//     gpuErrchk(cudaFree(d_db));
// }

// // Main function to mine patterns
// void mine_patterns(params p,
//                    const std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash> &filtered_transactions,
//                    std::vector<uint32_t> primary,
//                    std::vector<uint32_t> secondary,
//                    std::vector<pattern> &frequent_patterns,
//                    const std::unordered_map<uint32_t, std::string> &intToStr)
// {
//     auto total_start_time = std::chrono::high_resolution_clock::now();

//     // Prepare transactions
//     std::vector<uint32_t> transaction_start;
//     std::vector<uint32_t> transaction_end;
//     std::vector<key_value> item_utility;
//     uint32_t max_transaction_length = 0;

//     auto start = std::chrono::high_resolution_clock::now();

//     prepare_transactions(filtered_transactions, primary, secondary,
//                          transaction_start, transaction_end, item_utility,
//                          max_transaction_length);

//     std::cout << "Time to prepare transactions: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(
//                      std::chrono::high_resolution_clock::now() - start).count()
//               << "ms" << std::endl;

//     // Compute shared memory requirement
//     size_t shared_memory_requirement = compute_shared_memory_requirement(max_transaction_length);

//     // Check shared memory requirement
//     check_shared_memory_requirement(shared_memory_requirement);

//     // Copy transactions to GPU
//     start = std::chrono::high_resolution_clock::now();

//     database *d_db = nullptr;
//     size_t total_gpu_memory_used = copy_transactions_to_gpu(transaction_start, transaction_end, item_utility, d_db);

//     std::cout << "Time to copy transactions to GPU: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(
//                      std::chrono::high_resolution_clock::now() - start).count()
//               << "ms" << std::endl;

//     // Print GPU memory usage breakdown
//     // print_gpu_memory_usage(transaction_start, transaction_end, item_utility, total_gpu_memory_used);

//     // Perform mining on GPU
//     start = std::chrono::high_resolution_clock::now();

//     mine_gpu_patterns(p, d_db, primary, secondary, frequent_patterns, intToStr, shared_memory_requirement);

//     std::cout << "GPU Mining Time: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(
//                      std::chrono::high_resolution_clock::now() - start).count()
//               << "ms" << std::endl;

//     // Clean up GPU memory
//     free_gpu_memory(d_db);

//     std::cout << "Total Time: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(
//                      std::chrono::high_resolution_clock::now() - total_start_time).count()
//               << "ms" << std::endl;
// }
