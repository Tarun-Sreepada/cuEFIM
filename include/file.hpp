#pragma once
#include "../main.cuh"
#include "structure.cuh"

// struct raw_file
// {
//     char *data;
//     size_t size_bytes;
//     size_t submitted_bytes; // bytes submitted to io_uring
//     size_t retrieved_bytes; // bytes retrieved from io_uring
//     size_t processed_bytes; // bytes processed/parsed
//     std::vector<std::pair<size_t, size_t>> retrieved_indices;

//     int fd;

//     raw_file() : data(nullptr), size_bytes(0), processed_bytes(0), submitted_bytes(0), retrieved_bytes(0), fd(-1) {}
//     ~raw_file()
//     {
//         if (data != nullptr)
//             free(data);
//         if (fd != -1)
//             close(fd);
//     }

// };


// std::tuple<database, std::unordered_map<uint32_t,uint32_t>> read_file_cpu(results &r, params &p);