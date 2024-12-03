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
    std::vector<transaction> transactions;
};

