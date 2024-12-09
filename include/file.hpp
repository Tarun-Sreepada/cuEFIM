#pragma once
#include "../main.cuh"
#include "args.cuh"

struct raw_file
{
    char *data;
    size_t size_bytes;
    size_t submitted_bytes; // bytes submitted to io_uring
    size_t retrieved_bytes; // bytes retrieved from io_uring
    size_t processed_bytes; // bytes processed from data
    std::vector<std::pair<size_t, size_t>> retrieved_indices;

    int fd;

    raw_file() : data(nullptr), size_bytes(0), submitted_bytes(0), retrieved_bytes(0), processed_bytes(0), fd(-1) {}

    void close()
    {
        if (data != nullptr)
            free(data);
    }

};


raw_file read_file_cpu(results &r, Config::Params &params);