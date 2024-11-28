#include "mine.cuh"
#include "config.cuh"
#include <cuda_runtime.h>

__global__ void hello_world()
{
    printf("Hello World from GPU!\n");
}


std::vector<
    std::pair<
        std::vector<std::string>, uint64_t>>
generate_frequent_itemsets_cpu(
    const std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash> &filtered_transactions,
    const std::unordered_map<uint64_t, uint64_t> &subtree_util,
    const std::unordered_map<uint64_t, uint64_t> &secondary_util,
    uint64_t min_utility)
{
    std::cout << "Hello World from CPU!" << std::endl;
    return {};
}

std::vector<
    std::pair<
        std::vector<std::string>, uint64_t>>
generate_frequent_itemsets_gpu(
    const std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash> &filtered_transactions,
    const std::unordered_map<uint64_t, uint64_t> &subtree_util,
    const std::unordered_map<uint64_t, uint64_t> &secondary_util,
    uint64_t min_utility)
{

    hello_world<<<1, 1>>>();
    cudaDeviceSynchronize();

    return {};

}