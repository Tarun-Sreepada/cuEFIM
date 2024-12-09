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

template <typename T>
__global__ void print_array(T* array, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%u ", static_cast<uint32_t>(array[i]));  // Use static_cast for non-uint32_t types
    }
    printf("\n");
}

__global__ void print_key_value(key_value *array, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%u:%u ", array[i].key, array[i].value);
    }
    printf("\n");
}

__global__ void print_db(gpu_db *d_db)
{
    printf("Transaction Count: %lu\n", d_db->transaction_count);
    printf("Total Items: %lu\n", d_db->total_items);
    printf("Max Transaction Size: %lu\n", d_db->max_transaction_size);
    printf("Load Factor: %lu\n", d_db->load_factor);

    for (int i = 0; i < d_db->transaction_count; i++)
    {
        for (int j = d_db->csr_transaction_start.ptr()[i]; j < d_db->csr_transaction_end.ptr()[i]; j++)
        {
            printf("%u:%u ", d_db->compressed_spare_row_db.ptr()[j].key, d_db->compressed_spare_row_db.ptr()[j].value);
        }
        printf("\n");
    }
    printf("\n");


}


// Kernel to insert transactions into the hash table
__global__ void hash_transactions(gpu_db *d_db)

{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_db->transaction_count)
    {
        return;
    }
    uint32_t bucket_size = (d_db->csr_transaction_end.ptr()[tid] - d_db->csr_transaction_start.ptr()[tid]) * d_db->load_factor;
    uint32_t item_index_insert_start = d_db->csr_transaction_start.ptr()[tid] * d_db->load_factor;

    for (int i = d_db->csr_transaction_start.ptr()[tid]; i < d_db->csr_transaction_end.ptr()[tid]; i++)
    {
        uint32_t item = d_db->compressed_spare_row_db.ptr()[i].key;
        uint32_t hashIdx = hashFunction(item, bucket_size);

        // d_db->d_hashed_csr_db.ptr()
        while (true)
        {
            if (d_db->transaction_hash_db.ptr()[hashIdx + item_index_insert_start].key == 0)
            {
                d_db->transaction_hash_db.ptr()[hashIdx + item_index_insert_start].key = item;
                d_db->transaction_hash_db.ptr()[hashIdx + item_index_insert_start].value = i;
                break;
            }
            // Handle collisions (linear probing)
            hashIdx = (hashIdx + 1) % (bucket_size);
        }
    }
}

__global__ void print_hash_transactions(gpu_db *d_db)

{
    for (int i = 0; i < d_db->transaction_count; i++)
    {
        for (int j = d_db->csr_transaction_start.ptr()[i] * d_db->load_factor; j < d_db->csr_transaction_end.ptr()[i] * d_db->load_factor; j++)
        {
            printf("%u:%u ", d_db->transaction_hash_db.ptr()[j].key, d_db->transaction_hash_db.ptr()[j].value);
        }
        printf("\n");
    }
        printf("\n");
}

__device__ int64_t query_item(key_value *item_index, uint32_t start_search, uint32_t end_search, uint32_t item) {

    uint32_t tableSize = end_search - start_search;

    uint32_t hashIdx = hashFunction(item, tableSize);

    while (true) {
        if (item_index[hashIdx + start_search].key == 0) {
            return -1; // Item not found
        }
        if (item_index[hashIdx + start_search].key == item) {
            return item_index[hashIdx + start_search].value;
        }
        // Handle collisions (linear probing)
        hashIdx = (hashIdx + 1) % tableSize;
    }
}

__global__ void no_hash_table(gpu_db *d_db, workload *w)
{
    
}

__global__ void no_hash_table_shared_mem(gpu_db *d_db, workload *w)
{
    
}

__global__ void hash_table(gpu_db *d_db, workload *w)
{
    
}


extern __shared__ key_value shared_memory[];
__global__ void hash_table_shared_mem(gpu_db *d_db, workload *w)
{
    uint32_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t thread_id = threadIdx.x;

    if (block_id >= d_db->transaction_count || d_db->transaction_hits.ptr()[block_id] == 0)
    {
        return;
    }

    d_db->transaction_hits.ptr()[block_id] = 0;
    
    uint32_t transaction_start = d_db->csr_transaction_start.ptr()[block_id];
    uint32_t transaction_end = d_db->csr_transaction_end.ptr()[block_id];
    uint32_t transaction_length = transaction_end - transaction_start;

    for (uint32_t i = thread_id; i < transaction_length * d_db->load_factor; i += blockDim.x)
    {
        shared_memory[i].key = d_db->transaction_hash_db.ptr()[transaction_start * d_db->load_factor + i].key;
        shared_memory[i].value = d_db->transaction_hash_db.ptr()[transaction_start * d_db->load_factor + i].value;
    }

    __syncthreads();

    for (uint32_t i = thread_id; i < w->number_of_primaries; i += blockDim.x)
    {
        uint32_t curr_cand_util = 0;
        uint32_t curr_cand_hits = 0;
        int32_t location = -1;

        for (uint32_t j = 0; j < w->primary_size; j++)
        {
            uint32_t candidate = w->primary.ptr()[i * w->primary_size + j];
            location = query_item(shared_memory, 0, transaction_length * d_db->load_factor, candidate);
            if (location != -1)
            {
                curr_cand_hits++;
                curr_cand_util += d_db->compressed_spare_row_db.ptr()[location].value;
            } else break;
        }
        if (curr_cand_hits != w->primary_size)
        {
            continue;
        }

        d_db->transaction_hits.ptr()[block_id] = 1;
        atomicAdd(&w->primary_utility.ptr()[i], curr_cand_util);

        // calculate the TWU
        uint32_t ref = w->secondary_reference.ptr()[i];
        uint32_t secondary_index_start = w->number_of_secondaries * ref;

        // collect all utilities
        for (uint32_t j = location + 1; j < transaction_end; j++)
        {
            uint32_t item = d_db->compressed_spare_row_db.ptr()[j].key;
            if (w->secondary.ptr()[secondary_index_start + item]) // if the item is valid secondary
            {
                curr_cand_util += d_db->compressed_spare_row_db.ptr()[j].value;
            }
        }

        uint32_t temp = 0;

        uint32_t subtree_local_insert_location = i * w->number_of_secondaries;

        for (uint32_t j = location + 1; j < transaction_end; j++)
        {
            uint32_t item = d_db->compressed_spare_row_db.ptr()[j].key;
            if (w->secondary.ptr()[secondary_index_start + item]) // if the item is valid secondary
            {
                atomicAdd(&w->local_utility.ptr()[subtree_local_insert_location + item], curr_cand_util);
                atomicAdd(&w->subtree_utility.ptr()[subtree_local_insert_location + item], curr_cand_util - temp);
                temp += d_db->compressed_spare_row_db.ptr()[j].value;
            }
        }
    }

}

__global__ void clean_subtree_local_utility(gpu_db *db, workload *w, uint32_t minimum_utility)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= w->number_of_primaries)
    {
        return;
    }

    uint32_t subtree_local_insert_location = tid * w->number_of_secondaries;

    for (uint32_t i = 0; i < w->number_of_secondaries; i++)
    {
        if (w->subtree_utility.ptr()[subtree_local_insert_location + i] >= minimum_utility)
        {
            w->subtree_utility.ptr()[subtree_local_insert_location + i] = i;
            w->number_of_new_candidates_per_candidate.ptr()[tid + 1]++;
        }
        else
        {
            w->subtree_utility.ptr()[subtree_local_insert_location + i] = 0;
        }
        if (w->local_utility.ptr()[subtree_local_insert_location + i] >= minimum_utility)
        {
            w->local_utility.ptr()[subtree_local_insert_location + i] = i;
        }
        else
        {
            w->local_utility.ptr()[subtree_local_insert_location + i] = 0;
        }
    }
}

__global__ void create_new_candidates(workload *w)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= w->number_of_primaries)
    {
        return;
    }

    // if no new candidates
    if (w->number_of_new_candidates_per_candidate.ptr()[tid] == w->number_of_new_candidates_per_candidate.ptr()[tid + 1])
    {
        return;
    }

    uint32_t counter = w->primary_size * w->number_of_new_candidates_per_candidate.ptr()[tid];
    uint32_t refStart = w->number_of_new_candidates_per_candidate.ptr()[tid];

    for (uint32_t i = tid * w->number_of_secondaries; i < (tid + 1) * w->number_of_secondaries; i++)
    {
        if (w->subtree_utility.ptr()[i])
        {
            for (uint32_t j = tid * (w->primary_size - 1); j < (tid + 1) * (w->primary_size - 1); j++)
            {
                w->new_primaries.ptr()[counter] = w->primary.ptr()[j];
                counter++;
            }
            w->new_primaries.ptr()[counter] = w->subtree_utility.ptr()[i];
            counter++;
            w->new_secondary_reference.ptr()[refStart] = tid;
            refStart++;
        }
    }

    return;

}

void mine(build_file &bf, results &r, Config::Params &p)
{

    gpu_db *d_db;
    cudaMallocManaged(&d_db, sizeof(gpu_db));

    d_db->transaction_count = bf.transaction_count;
    d_db->total_items = bf.total_items;
    d_db->max_transaction_size = bf.max_transaction_size;

    d_db->compressed_spare_row_db = CudaMemory<key_value>(bf.compressed_spare_row_db, p.GPU_memory_allocation);
    d_db->csr_transaction_start = CudaMemory<size_t>(bf.csr_transaction_start, p.GPU_memory_allocation);
    d_db->csr_transaction_end = CudaMemory<size_t>(bf.csr_transaction_end, p.GPU_memory_allocation);
    d_db->transaction_hits = CudaMemory<bool>(bf.transaction_count, p.GPU_memory_allocation);
    // set transaciton hits all to true
    cudaMemset(d_db->transaction_hits.ptr(), 1, bf.transaction_count * sizeof(bool));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    r.record_memory_usage("GPU DB");

    dim3 block(1);
    dim3 grid(1);

    #ifdef DEBUG
    print_db<<<grid, block>>>(d_db);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif


    // if params is hash hash the thing
    if (p.method == Config::mine_method::hash_table || p.method == Config::mine_method::hash_table_shared_memory)
    {
        d_db->load_factor = 2;
        d_db->transaction_hash_db = CudaMemory<key_value>(d_db->total_items * d_db->load_factor, p.GPU_memory_allocation);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        block = dim3(p.block_size);
        grid = dim3((d_db->transaction_count + block.x - 1) / block.x);

        hash_transactions<<<grid, block>>>(d_db);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        r.record_memory_usage("Hash DB");

        #ifdef DEBUG
        print_hash_transactions<<<1, 1>>>(d_db);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        #endif
    }

    workload *w;
    cudaMallocManaged(&w, sizeof(workload));

    w->primary = CudaMemory<uint32_t>(bf.primary, p.GPU_memory_allocation);
    w->number_of_primaries = bf.primary.size();
    w->primary_size = 1;
    
    w->secondary_reference = CudaMemory<uint32_t>(bf.primary.size(), p.GPU_memory_allocation);
    cudaMemset(w->secondary_reference.ptr(), 0, bf.primary.size() * sizeof(uint32_t));

    w->secondary = CudaMemory<uint32_t>(bf.secondary, p.GPU_memory_allocation);
    w->number_of_secondaries = bf.secondary.size();

    grid = dim3(d_db->transaction_count); 


    while (w->number_of_primaries)
    {
        std::cout << "Primary size: " << w->primary_size << "\tNumber of primaries: " << w->number_of_primaries << std::endl;
        r.record_memory_usage("Iter: " + std::to_string(w->primary_size) + "a");

        w->primary_utility = CudaMemory<uint32_t>(w->number_of_primaries, p.GPU_memory_allocation);
        cudaMemset(w->primary_utility.ptr(), 0, w->number_of_primaries * sizeof(uint32_t));
        w->subtree_utility = CudaMemory<uint32_t>(w->number_of_primaries * w->number_of_secondaries, p.GPU_memory_allocation);
        cudaMemset(w->subtree_utility.ptr(), 0, w->number_of_primaries * w->number_of_secondaries * sizeof(uint32_t));
        w->local_utility = CudaMemory<uint32_t>(w->number_of_primaries * w->number_of_secondaries, p.GPU_memory_allocation);
        cudaMemset(w->local_utility.ptr(), 0, w->number_of_primaries * w->number_of_secondaries * sizeof(uint32_t));

        r.record_memory_usage("Iter: " + std::to_string(w->primary_size) + "b");

        // Call the search kernel
        grid = dim3(d_db->transaction_count);   
        if (p.method == Config::mine_method::no_hash_table)
        {
            no_hash_table<<<grid, block>>>(d_db, w);
        }
        else if (p.method == Config::mine_method::no_hash_table_shared_memory)
        {
            no_hash_table_shared_mem<<<grid, block>>>(d_db, w);
        }
        else if (p.method == Config::mine_method::hash_table)
        {
            hash_table<<<grid, block>>>(d_db, w);
        }
        else if (p.method == Config::mine_method::hash_table_shared_memory)
        {
            hash_table_shared_mem<<<grid, block>>>(d_db, w);
        }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // push the patterns to the vector
        r.patterns.push_back({w->primary.get(), w->primary_utility.get()});

        #ifdef DEBUG
        std::vector<uint32_t> primary = w->primary.get();
        std::vector<uint32_t> primary_utility = w->primary_utility.get();
        std::vector<uint32_t> subtree_utility = w->subtree_utility.get();
        std::vector<uint32_t> local_utility = w->local_utility.get();


        for (size_t i = 0; i < w->number_of_primaries; i++)
        {
            for (size_t j = 0; j < w->primary_size; j++)
            {
                std::cout << primary[i * w->primary_size + j] << " ";
            }
            std::cout << "#UTIL: " << primary_utility[i] << std::endl;
            std::cout << "Local utility: " << std::endl;
            for (size_t j = 0; j < w->number_of_secondaries; j++)
            {
                std::cout << local_utility[i * w->number_of_secondaries + j] << " ";
            }
            std::cout << std::endl;

            std::cout << "Subtree utility: " << std::endl;
            for (size_t j = 0; j < w->number_of_secondaries; j++)
            {
                std::cout << subtree_utility[i * w->number_of_secondaries + j] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
        
        #endif


        // clean up subtree and local utility
        w->number_of_new_candidates_per_candidate = CudaMemory<uint32_t>(w->number_of_primaries + 1, p.GPU_memory_allocation);
        cudaMemset(w->number_of_new_candidates_per_candidate.ptr(), 0, (w->number_of_primaries + 1) * sizeof(uint32_t));

        grid = dim3((w->number_of_primaries + p.block_size - 1) / p.block_size);
        clean_subtree_local_utility<<<grid, block>>>(d_db, w, p.min_utility);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // wrap using thrust 
        thrust::device_ptr<uint32_t> thrust_n_o_n_p_c = thrust::device_pointer_cast(w->number_of_new_candidates_per_candidate.ptr());
        w->total_number_new_primaries = thrust::reduce(thrust_n_o_n_p_c, thrust_n_o_n_p_c + w->number_of_primaries + 1);

        #ifdef DEBUG
        print_array<<<1, 1>>>(w->number_of_new_candidates_per_candidate.ptr(), w->number_of_primaries + 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        #endif

        r.record_memory_usage("Iter: " + std::to_string(w->primary_size) + "c");

        if (w->total_number_new_primaries == 0)
        {
            break;
        }

        thrust::inclusive_scan(thrust_n_o_n_p_c, thrust_n_o_n_p_c + w->number_of_primaries + 1, thrust_n_o_n_p_c);

        #ifdef DEBUG
        print_array<<<1, 1>>>(w->number_of_new_candidates_per_candidate.ptr(), w->number_of_primaries + 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        #endif


        w->primary_size += 1;
        w->new_primaries = CudaMemory<uint32_t>(w->total_number_new_primaries * w->primary_size, p.GPU_memory_allocation);
        w->new_secondary_reference = CudaMemory<uint32_t>(w->total_number_new_primaries, p.GPU_memory_allocation);

        create_new_candidates<<<grid, block>>>(w);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        #ifdef DEBUG
        std::vector<uint32_t> new_primaries = w->new_primaries.get();
        std::vector<uint32_t> new_secondary_reference = w->new_secondary_reference.get();

        for (size_t i = 0; i < w->total_number_new_primaries; i++)
        {
            for (size_t j = 0; j < w->primary_size; j++)
            {
                std::cout << new_primaries[i * w->primary_size + j] << " ";
            }
            std::cout << "#REF: " << new_secondary_reference[i] << std::endl;
        }
        #endif

        r.record_memory_usage("Iter: " + std::to_string(w->primary_size - 1) + "d");

        // move the new primaries to the primary
        w->primary = std::move(w->new_primaries);
        w->secondary_reference = std::move(w->new_secondary_reference);
        w->number_of_primaries = w->total_number_new_primaries;
        w->secondary = std::move(w->local_utility);

        r.record_memory_usage("Iter: " + std::to_string(w->primary_size - 1) + "e");

    }

    // free memory
    cudaFree(d_db);
    cudaFree(w);

}

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
