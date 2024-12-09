#include "convert.cuh"


CudaMemory<d_database> convert_to_gpu(build_file &bf, results &r, Config::Params &p)
{

    database db;
    CudaMemory<d_database> d_db(1, Config::gpu_memory_allocation::Unified);

    // primary 
    #ifdef DEBUG
    std::cout << "Primary: ";
    #endif
    for (auto &item : bf.subtree_utility)
    {
        if (item.second >= p.min_utility)
        {
            db.primary.push_back(item.first);
            #ifdef DEBUG
            std::cout << item.first << " ";
            #endif
        }
    }
    #ifdef DEBUG
    std::cout << std::endl;
    #endif

    // secondary
    // find max value of id in id2item 
    size_t max_id = std::max_element(bf.itemID_to_item.begin(), bf.itemID_to_item.end(), [](const auto &a, const auto &b) { return a.first < b.first; })->first;
    // db.secondary = // fill 1 to max_id  + 1
    db.secondary.assign(max_id + 1, 1);

    for (auto &transaction : bf.transactions)
    {
        db.csr_transaction_start.push_back(db.db_size);

        const auto &items = transaction.first;
        const auto &utils = transaction.second;
        for (int i = 0; i < items.size(); i++)
        {
            db.compressed_spare_row_db.push_back({items[i], utils[i]});
            #ifdef DEBUG
            std::cout << items[i] << ":" << utils[i] << " ";
            #endif
        }
        #ifdef DEBUG
        std::cout << std::endl;
        #endif
        db.db_size += items.size();

        db.max_transaction_size = std::max(db.max_transaction_size, items.size());

        db.csr_transaction_end.push_back(db.db_size);
    }

    db.csr_transaction_start.shrink_to_fit();
    db.csr_transaction_end.shrink_to_fit();
    db.compressed_spare_row_db.shrink_to_fit();

    r.record_memory_usage("Flattened DB");


        

    // d_db.d_compressed_spare_row_db = CudaMemory<key_value>(db.compressed_spare_row_db.size(), p.GPU_memory_allocation);
    // cudaMemcpy(d_db.d_compressed_spare_row_db.ptr(), db.compressed_spare_row_db.data(), db.compressed_spare_row_db.size() * sizeof(key_value), cudaMemcpyHostToDevice);
    // d_db.d_csr_transaction_start = CudaMemory<size_t>(db.csr_transaction_start.size(), p.GPU_memory_allocation);
    // d_db.d_csr_transaction_end = CudaMemory<size_t>(db.csr_transaction_end.size(), p.GPU_memory_allocation);

    // cudaMemcpy(d_db.d_csr_transaction_start.ptr(), db.csr_transaction_start.data(), db.csr_transaction_start.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_db.d_csr_transaction_end.ptr(), db.csr_transaction_end.data(), db.csr_transaction_end.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    // d_db.primary = CudaMemory<uint32_t>(db.primary.size(), p.GPU_memory_allocation);
    // d_db.secondary = CudaMemory<uint32_t>(db.secondary.size(), p.GPU_memory_allocation);

    // cudaMemcpy(d_db.primary.ptr(), db.primary.data(), db.primary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_db.secondary.ptr(), db.secondary.data(), db.secondary.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    // d_db.primary_size = db.primary.size();
    // d_db.secondary_size = db.secondary.size();

    // d_db.valid_transaction = CudaMemory<bool>(db.db_size, p.GPU_memory_allocation);
    // // set all to true

    // cudaMemset(d_db.valid_transaction.ptr(), 1, db.db_size * sizeof(bool));

    // if (p.method == Config::mine_method::hash_table || p.method == Config::mine_method::hash_table_shared_memory)
    // {
    //     d_db.load_factor = 2;
    //     d_db.d_hashed_csr_db = CudaMemory<key_value>(d_db.load_factor * db.db_size, p.GPU_memory_allocation);

    // }

    // d_db.largest_transaction_size = db.max_transaction_size;
    // d_db.transaction_count = db.csr_transaction_start.size();

    // r.record_memory_usage("To GPU");


    return d_db;

}