#include "mine.cuh"
#include "config.cuh"
#include "database.cuh"

__global__ void hello_world()
{
    printf("Hello World from GPU!\n");
}

// searchGPU<<<grid, block>>>(d_db, thrust::raw_pointer_cast(transaction_hits.data()), transactions_count,
//                                     thrust::raw_pointer_cast(d_candidates.data()), number_of_candidates, candidate_size,
//                                     thrust::raw_pointer_cast(d_secondary.data()), secondary_size,
//                                     thrust::raw_pointer_cast(d_secondary_reference.data()),
//                                     thrust::raw_pointer_cast(d_candidate_utility.data()),
//                                     thrust::raw_pointer_cast(d_candidate_subtree_utility.data()),
//                                     thrust::raw_pointer_cast(d_candidate_local_utility.data()));

__global__ void print_array(uint32_t *array, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++)
    {
        printf("%u ", array[i]);
    }
    printf("\n");
}


extern __shared__ key_value shared_memory[];
__global__ void searchGPU(database *d_db, uint32_t *transaction_hits, uint32_t transactions_count,
                          uint32_t *candidates, uint32_t number_of_candidates, uint32_t candidate_size,
                          uint32_t *secondary, uint32_t secondary_size,
                          uint32_t *secondary_reference,
                          uint32_t *candidate_utility,
                          uint32_t *candidate_subtree_utility,
                          uint32_t *candidate_local_utility)
{
    uint32_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t tid = threadIdx.x;
    if (block_id >= transactions_count || transaction_hits[block_id] == 0)
    {
        return;
    }
    transaction_hits[block_id] = 0;

    uint32_t transaction_start = d_db->transaction_start[block_id];
    uint32_t transaction_end = d_db->transaction_end[block_id];
    uint32_t transaction_length = transaction_end - transaction_start;

    for (uint32_t i = tid; i < transaction_length * bucket_factor; i += blockDim.x)
    {
        // shared_memory[i] = d_db->item_index[transaction_start * bucket_factor + i];
        shared_memory[i].key = d_db->item_index[transaction_start * bucket_factor + i].key;
        shared_memory[i].value = d_db->item_index[transaction_start * bucket_factor + i].value;
    }   

    __syncthreads();

    uint32_t curr_cand_hits = 0;
    int32_t location = -1;
    uint32_t curr_cand_util = 0;
    for (uint32_t i = tid; i < number_of_candidates; i += blockDim.x)
    {
        for (uint32_t j = 0; j < candidate_size; j++)
        {
            uint32_t candidate = candidates[i * candidate_size + j];
            location = query_item(shared_memory, 0, transaction_length * bucket_factor, candidate);
            if (location != -1)
            {
                curr_cand_hits++;
                // curr_cand_util += shared_memory[location].value;
                curr_cand_util += d_db->item_utility[location].value;
            }
        }
        if (curr_cand_hits != candidate_size)
        {
            continue;
        }
        transaction_hits[block_id] += 1;
        atomicAdd(&candidate_utility[i], curr_cand_util);

        uint32_t all_utilities = curr_cand_util;
        location -= transaction_start;

        // collect all utilities
        for (uint32_t j = location + 1; j < transaction_length; j++)
        {
            uint64_t new_loc = j + transaction_start;
            uint32_t item = d_db->item_utility[new_loc].key;
            // i->secondary->ref->secondary_size*ref->secondary + item 
            uint32_t ref = secondary_reference[i];
            uint32_t secondary_index = secondary_size * ref + item - 1;
            // printf("ref: %u, secondary size: %u, secondary index: %u\n", ref, secondary_size, secondary_index);
            // item ref, secondary_size, secondary_size * ref, item
            // printf("item: %u, ref: %u, secondary_size: %u, secondary_index: %u\n", item, ref, secondary_size, secondary_index);
            if (secondary[secondary_index]) // if the item is valid secondary
            {
                all_utilities += d_db->item_utility[new_loc].value;
            }
        }

        uint32_t temp = 0;
        // calculate local and subtree utility
        for (uint32_t j = location + 1; j < transaction_length; j++)
        {
            uint64_t new_loc = j + transaction_start;
            uint32_t item = d_db->item_utility[new_loc].key;
            // i->secondary->ref->secondary_size*ref->secondary + item 
            uint32_t ref = secondary_reference[i];
            uint32_t secondary_index = secondary_size * ref + item - 1;
            if (secondary[secondary_index]) // if the item is valid secondary
            {
                // candidate di * secondary size + item - 1
                uint32_t secondary_index = secondary_size * i + item - 1;
                atomicAdd(&candidate_local_utility[secondary_index], all_utilities - temp);
                atomicAdd(&candidate_subtree_utility[secondary_index], all_utilities - temp);
                temp += d_db->item_utility[new_loc].value;
            }
        }


        // we have collect the utility and the last location of the last candidate
        // add up all the utilities of values in the transaction which are in secondary using secondary  ref and also primary
    }


}


__global__ void clean_subtree_local_utility(uint32_t number_of_candidates, uint32_t *number_of_new_candidates_per_candidate, 
                                            uint32_t *subtree_utility, uint32_t *local_utility, uint32_t secondary_size, uint32_t minimum_utility)
{
    uint32_t tid = threadIdx.x;
    if (tid >= number_of_candidates)
    {
        return;
    }

    for (uint32_t i = tid * secondary_size; i < (tid + 1) * secondary_size; i++)
    {
        if (subtree_utility[i] >= minimum_utility)
        {
            subtree_utility[i] = i - tid * secondary_size + 1;
            number_of_new_candidates_per_candidate[tid + 1]++;
        }
        else
        {
            subtree_utility[i] = 0;
        }
        if (local_utility[i] >= minimum_utility)
        {
            local_utility[i] = i - tid * secondary_size + 1;
        }
        else
        {
            local_utility[i] = 0;
        }
    }
    return;
}

// create_new_candidates<<<1, 1>>>(thrust::raw_pointer_cast(d_candidates.data()), thrust::raw_pointer_cast(d_candidate_subtree_utility.data()), 
//                                         number_of_candidates,thrust::raw_pointer_cast(d_new_candidates.data()), 
//                                         thrust::raw_pointer_cast(d_new_secondary_reference.data()), secondary_size, candidate_size, 
//                                         thrust::raw_pointer_cast(d_number_of_new_candidates_per_candidate.data()));


__global__ void create_new_candidates(uint32_t *candidates, uint32_t *candidate_subtree_utility, uint32_t number_of_candidates,
                                      uint32_t *new_candidates, uint32_t *new_secondary_reference, uint32_t secondary_size, uint32_t candidate_size,
                                      uint32_t *number_of_new_candidates_per_candidate)
{
    uint32_t tid = threadIdx.x;
    if (tid >= number_of_candidates)
    {
        return;
    }

    if (number_of_new_candidates_per_candidate[tid] == number_of_new_candidates_per_candidate[tid + 1])
    {
        return;
    }

    uint32_t counter = candidate_size * number_of_new_candidates_per_candidate[tid];
    uint32_t refStart = number_of_new_candidates_per_candidate[tid];
    for (uint32_t i = tid * secondary_size; i < (tid + 1) * secondary_size; i++)
    {
        if (candidate_subtree_utility[i])
        {
            for (uint32_t j = tid * (candidate_size - 1); j < (candidate_size + 1) * (candidate_size - 1); j++)
            {
                new_candidates[counter] = candidates[j];
                counter++;
            }
            new_candidates[counter] = candidate_subtree_utility[i];
            counter++;
            new_secondary_reference[refStart] = tid;
            refStart++;
        }
    }

    return;
}

void mine_patterns(params p, std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash> filtered_transactions,
                   std::vector<uint32_t> primary, std::vector<uint32_t> secondary,
                   std::vector<pattern> frequent_patterns)
{
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t max_transaction_length = 0;
    for (const auto &transaction : filtered_transactions)
    {
        max_transaction_length = std::max(max_transaction_length, (uint32_t)transaction.first.size());
    }

    std::cout << "Max transaction length: " << max_transaction_length << std::endl;
    std::cout << "Number of transactions: " << filtered_transactions.size() << std::endl;


    // sort secondary
    std::sort(secondary.begin(), secondary.end());
    // print_vector(secondary);
    for (uint32_t i = 0; i < secondary.size(); i++)
    {
        std::cout << secondary[i] << " ";
    }
    std::cout << std::endl;

    std::vector<uint32_t> transaction_start;
    std::vector<uint32_t> transaction_end;
    std::vector<key_value> item_utility;

    for (const auto &transaction : filtered_transactions)
    {
        transaction_start.push_back(item_utility.size());
        for (uint32_t i = 0; i < transaction.first.size(); i++)
        {
            item_utility.push_back({transaction.first[i], transaction.second[i]});
        }
        transaction_end.push_back(item_utility.size());
    }


    std::cout << "Time to convert transactions: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    uint32_t shared_memory_requirement = max_transaction_length * sizeof(key_value) * bucket_factor * 2; // twice as much just to be safe // tweak later
    std::cout << "Shared memory requirement: " << shared_memory_requirement * sizeof(key_value) << std::endl;
    // query the device for the maximum shared memory per block
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    std::cout << "Max shared memory per block: " << props.sharedMemPerBlock << std::endl;

    if (shared_memory_requirement > props.sharedMemPerBlock)
    {
        std::cerr << "Shared memory requirement exceeds the maximum shared memory per block" << std::endl;
    }
    
    std::cout << "Time to convert transactions: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // Create the database
    database *d_db;
    gpuErrchk(cudaMalloc(&d_db, sizeof(database)));

    uint32_t *d_transaction_start;
    uint32_t *d_transaction_end;

    key_value *d_item_utility;
    key_value *d_item_index;

    gpuErrchk(cudaMalloc(&d_transaction_start, transaction_start.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_transaction_end, transaction_end.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_item_utility, item_utility.size() * sizeof(key_value)));
    gpuErrchk(cudaMalloc(&d_item_index, item_utility.size() * bucket_factor * sizeof(key_value)));

    gpuErrchk(cudaMemcpy(d_transaction_start, transaction_start.data(), transaction_start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transaction_end, transaction_end.data(), transaction_end.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_item_utility, item_utility.data(), item_utility.size() * sizeof(key_value), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_item_index, 0, item_utility.size() * bucket_factor * sizeof(key_value)));

    uint32_t transactions_count = filtered_transactions.size();
    cudaMemcpy(&d_db->transactions_count, &transactions_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_db->transaction_start, &d_transaction_start, sizeof(uint32_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_db->transaction_end, &d_transaction_end, sizeof(uint32_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_db->item_utility, &d_item_utility, sizeof(key_value *), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_db->item_index, &d_item_index, sizeof(key_value *), cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // print_array<<<1, 1>>>(d_transaction_start, transaction_start.size());
    // gpuErrchk(cudaDeviceSynchronize());

    std::cout << "Time to copy transactions to GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // print_db<<<1, 1>>>(d_db);
    // gpuErrchk(cudaDeviceSynchronize());
    // gpuErrchk(cudaPeekAtLastError());

    // Call the kernel
    dim3 block(block_size);
    dim3 grid((transactions_count + block.x - 1) / block.x);

    hash_transactions<<<grid, block>>>(d_db); // each thread will handle a transaction
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "Time to hash transactions: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    uint32_t number_of_candidates = primary.size();
    uint32_t candidate_size = 1;

    print_db_full<<<1, 1>>>(d_db);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());


    thrust::device_vector<uint32_t> d_candidates = primary;
    thrust::device_vector<uint32_t> d_secondary_reference(primary.size(), 0);
    thrust::device_vector<uint32_t> d_secondary = secondary;

    thrust::sort(d_secondary.begin(), d_secondary.end());
    uint32_t secondary_size = d_secondary.size();
    thrust::device_vector<uint32_t> transaction_hits(transactions_count, 1);

    std::vector<std::pair<thrust::host_vector<uint32_t>, thrust::host_vector<uint32_t>>> original_patterns;


    while (number_of_candidates)
    {
        std::cout << "Number of candidates: " << number_of_candidates << std::endl;

        thrust::device_vector<uint32_t> d_candidate_utility(number_of_candidates, 0);
        thrust::device_vector<uint32_t> d_candidate_subtree_utility(number_of_candidates * secondary_size, 0);
        thrust::device_vector<uint32_t> d_candidate_local_utility(number_of_candidates * secondary_size, 0);

        // block size is 32 but grid is number of transactions
        block = dim3(block_size, 1, 1);
        grid = dim3(transactions_count, 1, 1);

        searchGPU<<<grid, block, shared_memory_requirement>>>(d_db, thrust::raw_pointer_cast(transaction_hits.data()), transactions_count,
                                    thrust::raw_pointer_cast(d_candidates.data()), number_of_candidates, candidate_size,
                                    thrust::raw_pointer_cast(d_secondary.data()), secondary_size,
                                    thrust::raw_pointer_cast(d_secondary_reference.data()),
                                    thrust::raw_pointer_cast(d_candidate_utility.data()),
                                    thrust::raw_pointer_cast(d_candidate_subtree_utility.data()),
                                    thrust::raw_pointer_cast(d_candidate_local_utility.data()));

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        thrust::host_vector<uint32_t> h_candidates = d_candidates;
        thrust::host_vector<uint32_t> h_candidate_utility = d_candidate_utility;

        original_patterns.push_back({h_candidates, h_candidate_utility});

        // clean subtree local utility
        candidate_size += 1;

        thrust::device_vector<uint32_t> d_number_of_new_candidates_per_candidate(number_of_candidates, 0);

        clean_subtree_local_utility<<<1, 1>>>(number_of_candidates, thrust::raw_pointer_cast(d_number_of_new_candidates_per_candidate.data()), 
                                            thrust::raw_pointer_cast(d_candidate_subtree_utility.data()), thrust::raw_pointer_cast(d_candidate_local_utility.data()), 
                                            secondary_size, p.min_utility);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        uint32_t number_of_new_candidates = thrust::reduce(d_number_of_new_candidates_per_candidate.begin(), d_number_of_new_candidates_per_candidate.end());
        thrust::inclusive_scan(d_number_of_new_candidates_per_candidate.begin(), d_number_of_new_candidates_per_candidate.end(), d_number_of_new_candidates_per_candidate.begin());

        if (number_of_new_candidates == 0)
        {
            break;
        }

        thrust::device_vector<uint32_t> d_new_candidates(number_of_new_candidates, 0);
        thrust::device_vector<uint32_t> d_new_secondary_reference(number_of_new_candidates, 0);

        // generate new candidates
            //    createNewCands<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(cands.data()), thrust::raw_pointer_cast(subtreeUtils.data()), numCands,
            //                                    thrust::raw_pointer_cast(newCands.data()), thrust::raw_pointer_cast(newSecondaryRefs.data()),
            //  numSecondary, candSize, thrust::raw_pointer_cast(numNewCandsPerCand.data()));

        create_new_candidates<<<1, 1>>>(thrust::raw_pointer_cast(d_candidates.data()), thrust::raw_pointer_cast(d_candidate_subtree_utility.data()), 
                                        number_of_candidates,thrust::raw_pointer_cast(d_new_candidates.data()), 
                                        thrust::raw_pointer_cast(d_new_secondary_reference.data()), secondary_size, candidate_size, 
                                        thrust::raw_pointer_cast(d_number_of_new_candidates_per_candidate.data()));


        number_of_candidates = number_of_new_candidates;
        d_candidates = d_new_candidates;
        d_secondary_reference = d_new_secondary_reference;

    }

    uint32_t pattern_counter = 0;
    for (const auto &pattern : original_patterns)
    {
        thrust::host_vector<uint32_t> h_candidates = pattern.first;
        thrust::host_vector<uint32_t> h_candidate_utility = pattern.second;

        for (auto &util : h_candidate_utility)
        {
            if (util >= p.min_utility)
            {
                pattern_counter++;
            }
        }
    }
    std::cout << "Number of patterns: " << pattern_counter << std::endl;

}