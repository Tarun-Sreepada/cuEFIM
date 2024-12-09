#pragma once
#include "../main.cuh"
#include "memory.cuh"

struct database
{
    std::vector<key_value> compressed_spare_row_db;
    std::vector<size_t> csr_transaction_start;
    std::vector<size_t> csr_transaction_end;
    size_t db_size;
    size_t max_transaction_size;

    std::vector<uint32_t> primary;
    std::vector<uint32_t> secondary;


    database()
    {
        db_size = 0;
        max_transaction_size = 0;
    }

    ~database()
    {
        compressed_spare_row_db.clear();
        csr_transaction_start.clear();
        csr_transaction_end.clear();

    }
};

