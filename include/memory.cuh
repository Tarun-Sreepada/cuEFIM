#pragma once
#include "../main.cuh"

uint32_t used_gpu_memory()
{
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    return total_byte - free_byte;
}