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
__global__ void print_array(T *array, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%u ", static_cast<uint32_t>(array[i])); // Use static_cast for non-uint32_t types
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

__device__ int64_t query_item(key_value *item_index, uint32_t start_search, uint32_t end_search, uint32_t item)
{

    uint32_t tableSize = end_search - start_search;

    uint32_t hashIdx = hashFunction(item, tableSize);

    while (true)
    {
        if (item_index[hashIdx + start_search].key == 0)
        {
            return -1; // Item not found
        }
        if (item_index[hashIdx + start_search].key == item)
        {
            return item_index[hashIdx + start_search].value;
        }
        // Handle collisions (linear probing)
        hashIdx = (hashIdx + 1) % tableSize;
    }
}

__device__ int binary_search(key_value *item_index, uint32_t start_search, uint32_t end_search, uint32_t item)
{

    uint32_t left = start_search;
    uint32_t right = end_search - 1;

    if (item_index[left].key == item)
    {
        return left;
    }
    if (item_index[right].key == item)
    {
        return right;
    }
    if (item_index[left].key > item || item_index[right].key < item)
    {
        return -1;
    }

    while (left <= right)
    {
        uint32_t mid = left + (right - left) / 2;

        if (item_index[mid].key == item)
        {
            return mid; // Item found
        }

        if (item_index[mid].key < item)
        {
            left = mid + 1; // Move right
        }
        else
        {
            right = mid - 1; // Move left
        }
    }

    return -1; // Item not found
}

__global__ void no_hash_table(gpu_db *d_db, workload *w)
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

    for (uint32_t i = thread_id; i < w->number_of_primaries; i += blockDim.x)
    {
        uint32_t curr_cand_util = 0;
        uint32_t curr_cand_hits = 0;
        int32_t location = -1;

        for (uint32_t j = 0; j < w->primary_size; j++)
        {
            uint32_t candidate = w->primary.ptr()[i * w->primary_size + j];

            // // location = binary_search(shared_memory, 0, transaction_length, candidate);
            location = binary_search(d_db->compressed_spare_row_db.ptr(), transaction_start, transaction_end, candidate);

            // for (int k = transaction_start; k < transaction_end; k++)
            // {
            //     if (d_db->compressed_spare_row_db.ptr()[k].key == candidate)
            //     {
            //         location = k;
            //         break;
            //     }
            // }

            if (location != -1 && d_db->compressed_spare_row_db.ptr()[location].key == candidate)
            {
                curr_cand_hits++;
                curr_cand_util += d_db->compressed_spare_row_db.ptr()[location].value;
            }
            else
                break;
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

extern __shared__ key_value shared_memory[];
__global__ void no_hash_table_shared_mem(gpu_db *d_db, workload *w)
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

    for (uint32_t i = thread_id; i < transaction_length; i += blockDim.x)
    {
        shared_memory[i].key = d_db->compressed_spare_row_db.ptr()[transaction_start + i].key;
        shared_memory[i].value = d_db->compressed_spare_row_db.ptr()[transaction_start + i].value;
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

            location = binary_search(shared_memory, 0, transaction_length, candidate);
            // for (int k = 0; k < transaction_length; k++)
            // {
            //     if (shared_memory[k].key == candidate)
            //     {
            //         location = k;
            //         break;
            //     }
            // }

            if (location != -1 && shared_memory[location].key == candidate)
            {
                curr_cand_hits++;
                curr_cand_util += shared_memory[location].value;
            }
            else
                break;
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
        for (uint32_t j = location + 1; j < transaction_length; j++)
        {
            // uint32_t item = d_db->compressed_spare_row_db.ptr()[j].key;
            uint32_t item = shared_memory[j].key;
            if (w->secondary.ptr()[secondary_index_start + item]) // if the item is valid secondary
            {
                curr_cand_util += shared_memory[j].value;
            }
        }

        uint32_t temp = 0;

        uint32_t subtree_local_insert_location = i * w->number_of_secondaries;

        for (uint32_t j = location + 1; j < transaction_length; j++)
        {
            uint32_t item = shared_memory[j].key;

            if (w->secondary.ptr()[secondary_index_start + item]) // if the item is valid secondary
            {
                atomicAdd(&w->local_utility.ptr()[subtree_local_insert_location + item], curr_cand_util);
                atomicAdd(&w->subtree_utility.ptr()[subtree_local_insert_location + item], curr_cand_util - temp);
                temp += shared_memory[j].value;
            }
        }
    }
}

__global__ void hash_table(gpu_db *d_db, workload *w)
{

    uint32_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t thread_id = threadIdx.x;

    if (block_id >= d_db->transaction_count || d_db->transaction_hits.ptr()[block_id] == 0)
    {
        return;
    }

    uint32_t transaction_start = d_db->csr_transaction_start.ptr()[block_id];
    uint32_t transaction_end = d_db->csr_transaction_end.ptr()[block_id];

    for (uint32_t i = thread_id; i < w->number_of_primaries; i += blockDim.x)
    {
        uint32_t curr_cand_util = 0;
        uint32_t curr_cand_hits = 0;
        int32_t location = -1;

        for (uint32_t j = 0; j < w->primary_size; j++)
        {
            uint32_t candidate = w->primary.ptr()[i * w->primary_size + j];
            location = query_item(d_db->transaction_hash_db.ptr(), transaction_start * d_db->load_factor, transaction_end * d_db->load_factor, candidate);
            if (location != -1)
            {
                curr_cand_hits++;
                curr_cand_util += d_db->compressed_spare_row_db.ptr()[location].value;
            }
            else
                break;
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
            }
            else
                break;
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
    uint32_t primary_count = w->number_of_primaries;

    if (tid >= primary_count)
        return;


    for (uint32_t i = tid * w->number_of_secondaries; i < (tid + 1) * w->number_of_secondaries; i++)
    {
        uint32_t item_value = i - tid * w->number_of_secondaries;
        if (w->subtree_utility.ptr()[i] >= minimum_utility)
        {
            w->subtree_utility.ptr()[i] = item_value;
            w->number_of_new_candidates_per_candidate.ptr()[tid + 1]++;
        }
        else
        {
            w->subtree_utility.ptr()[i] = 0;
        }
        if (w->local_utility.ptr()[i] >= minimum_utility)
        {
            w->local_utility.ptr()[i] = item_value;
        }
        else
        {
            w->local_utility.ptr()[i] = 0;
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

    size_t shared_memory_required = d_db->max_transaction_size * sizeof(key_value);
    if (p.method == Config::mine_method::hash_table_shared_memory)
    {
        shared_memory_required *= d_db->load_factor;
    }

    #ifdef DEBUG
    print_db<<<1, 1>>>(d_db);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    while (w->number_of_primaries)
    {
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
            no_hash_table_shared_mem<<<grid, block, shared_memory_required>>>(d_db, w);
        }
        else if (p.method == Config::mine_method::hash_table)
        {
            hash_table<<<grid, block>>>(d_db, w);
        }
        else if (p.method == Config::mine_method::hash_table_shared_memory)
        {
            hash_table_shared_mem<<<grid, block, shared_memory_required>>>(d_db, w);
        }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        #ifdef DEBUG
        std::vector<uint32_t> primary = w->primary.get();
        std::vector<uint32_t> primary_utility = w->primary_utility.get();
        std::vector<uint32_t> subtree_utility = w->subtree_utility.get();
        std::vector<uint32_t> local_utility = w->local_utility.get();

        for (int i = 0; i < w->number_of_primaries; i++)
        {
            for (int j = 0; j < w->primary_size; j++)
            {
                std::cout << primary[i * w->primary_size + j] << " ";
            }
            std::cout << "Utility: " << primary_utility[i] << std::endl;
            for (int j = 0; j < w->number_of_secondaries; j++)
            {
                std::cout << "Subtree: " << subtree_utility[i * w->number_of_secondaries + j] << " ";
                std::cout << "Local: " << local_utility[i * w->number_of_secondaries + j] << std::endl;
            }
            std::cout << std::endl;
        }

        #endif

        // push the patterns to the vector
        r.patterns.push_back({w->primary.get(), w->primary_utility.get()});

        // clean up subtree and local utility
        w->number_of_new_candidates_per_candidate = CudaMemory<uint32_t>(w->number_of_primaries + 1, p.GPU_memory_allocation);
        cudaMemset(w->number_of_new_candidates_per_candidate.ptr(), 0, (w->number_of_primaries + 1) * sizeof(uint32_t));

        block = dim3(p.block_size);
        grid = dim3((w->number_of_primaries + p.block_size - 1) / p.block_size);
        clean_subtree_local_utility<<<grid, block>>>(d_db, w, p.min_utility);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        #ifdef DEBUG
        subtree_utility = w->subtree_utility.get();
        local_utility = w->local_utility.get();

        for (int i = 0; i < w->number_of_primaries; i++)
        {
            for (int j = 0; j < w->primary_size; j++)
            {
                std::cout << primary[i * w->primary_size + j] << " ";
            }
            std::cout << "Utility: " << primary_utility[i] << std::endl;
            for (int j = 0; j < w->number_of_secondaries; j++)
            {
                std::cout << "Subtree: " << subtree_utility[i * w->number_of_secondaries + j] << " ";
                std::cout << "Local: " << local_utility[i * w->number_of_secondaries + j] << std::endl;
            }
            std::cout << std::endl;
        }

        #endif

        // wrap using thrust
        thrust::device_ptr<uint32_t> thrust_n_o_n_p_c = thrust::device_pointer_cast(w->number_of_new_candidates_per_candidate.ptr());
        w->total_number_new_primaries = thrust::reduce(thrust_n_o_n_p_c, thrust_n_o_n_p_c + w->number_of_primaries + 1);
        #ifdef DEBUG
        std::cout << "Total number of new primaries: " << w->total_number_new_primaries << std::endl;
        #endif

        r.record_memory_usage("Iter: " + std::to_string(w->primary_size) + "c");

        if (w->total_number_new_primaries == 0)
        {
            break;
        }

        thrust::inclusive_scan(thrust_n_o_n_p_c, thrust_n_o_n_p_c + w->number_of_primaries + 1, thrust_n_o_n_p_c);

        w->primary_size += 1;
        w->new_primaries = CudaMemory<uint32_t>(w->total_number_new_primaries * w->primary_size, p.GPU_memory_allocation);
        w->new_secondary_reference = CudaMemory<uint32_t>(w->total_number_new_primaries, p.GPU_memory_allocation);

        create_new_candidates<<<grid, block>>>(w);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

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
