#pragma once
#include "../main.cuh"

struct key_value
{
    uint32_t key;
    uint32_t value;
};

struct transaction
{
    std::vector<key_value> item_utility;
};

struct database
{
    std::vector<key_value> compressed_spare_row_db;
    std::vector<size_t> csr_transaction_start;
    std::vector<size_t> csr_transaction_end;

    database()
    {
    }

    ~database()
    {
        compressed_spare_row_db.clear();
        csr_transaction_start.clear();
        csr_transaction_end.clear();
    }
};
