// #include "memory.cuh"

// void MemoryUtils::getMemoryInfo(size_t &freeMemory, size_t &usedMemory) {
//     size_t totalMemory;
//     cudaError_t err = cudaMemGetInfo(&freeMemory, &totalMemory);
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
//         freeMemory = 0;
//         usedMemory = 0;
//         return;
//     }
//     usedMemory = totalMemory - freeMemory;
// }

// void MemoryUtils::printMemoryInfo() {
//     size_t freeMemory, usedMemory;
//     getMemoryInfo(freeMemory, usedMemory);

//     std::cout << "==================== GPU Memory Info ====================" << std::endl;
//     std::cout << "Free Memory: " << freeMemory / (1024 * 1024) << " MB" << std::endl;
//     std::cout << "Used Memory: " << usedMemory / (1024 * 1024) << " MB" << std::endl;
//     std::cout << "=========================================================" << std::endl;
// }