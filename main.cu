// #include  "main.cuh"
// #include "include/args.cuh"
// #include "include/memory.cuh"
// #include "include/file.hpp"
// #include "include/build.hpp"
// #include "include/mine.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <string>


int main(int argc, char *argv[]) {
    
    // set device to 0
    cudaError_t ret = cudaSetDevice(0);
    if (ret != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(ret) << std::endl;
    }

    size_t free_mem, total_mem;

    
    ret = cudaMemGetInfo(&free_mem, &total_mem);
    if (ret != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(ret) << std::endl;
    }
    std::cout << "Free memory: " << free_mem << " bytes, Total memory: " << total_mem << " bytes\n";


    // auto params = Config::parse_arguments(argc, argv);
    // Config::print_arguments(params);
    // // set device
    // Config::set_device(params.cuda_device_id);

    // Config::print_gpu_stats();

    // results r;
    // // r.get_cuda_memory_usage();
    // std::cout << "GPU Memory Usage: " << r.get_cuda_memory_usage() << std::endl;
    
    // MemoryUtils::printMemoryInfo();

    // read file
    // read_file(r,p);

    // convert file to database using read file params

    // convert to appropriate structure for GPU mining

    // mine

    // write to file


    return 0;

}   