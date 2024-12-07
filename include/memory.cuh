#pragma once
#include "../main.cuh"
#include <cuda_runtime.h>

// class MemoryUtils {
// public:
//     // Function to get memory info on the GPU
//     static void getMemoryInfo(size_t &freeMemory, size_t &usedMemory);

//     // Function to print memory info
//     static void printMemoryInfo();
// };

// template <typename T>
// class CudaMemory {
// public:
//     // Constructor with optional allocation
//     CudaMemory(size_t size = 0, gpu_memory_allocation memType = gpu_memory_allocation::Device)
//         : size_(size), ptr_(nullptr) {
//         if (size > 0) {
//             allocateMemory(memType);
//         }
//     }

//     // Destructor
//     ~CudaMemory() {
//         deallocateMemory();
//     }

//     // Copy constructor
//     CudaMemory(const CudaMemory& other)
//         : size_(other.size_), ptr_(nullptr) {
//         if (size_ > 0) {
//             allocateMemory(gpu_memory_allocation::Device);
//             cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
//         }
//     }

//     // Copy assignment operator
//     CudaMemory& operator=(const CudaMemory& other) {
//         if (this != &other) {
//             deallocateMemory();
//             size_ = other.size_;
//             if (size_ > 0) {
//                 allocateMemory(gpu_memory_allocation::Device);
//                 cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
//             }
//         }
//         return *this;
//     }

//     // Move constructor
//     CudaMemory(CudaMemory&& other) noexcept
//         : size_(other.size_), ptr_(other.ptr_) {
//         other.ptr_ = nullptr;
//         other.size_ = 0;
//     }

//     // Move assignment operator
//     CudaMemory& operator=(CudaMemory&& other) noexcept {
//         if (this != &other) {
//             deallocateMemory();
//             size_ = other.size_;
//             ptr_ = other.ptr_;
//             other.ptr_ = nullptr;
//             other.size_ = 0;
//         }
//         return *this;
//     }

//     // Accessor for the GPU pointer
//     T* ptr() const { return ptr_; }

//     // Retrieve data from the GPU to the host
//     std::vector<T> get() const {
//         if (!ptr_) {
//             return {};
//         }
//         std::vector<T> h_vec(size_);
//         cudaMemcpy(h_vec.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
//         return h_vec;
//     }

// private:
//     size_t size_;
//     T* ptr_;

//     // Allocate memory
//     void allocateMemory(gpu_memory_allocation memType) {
//         if (memType == gpu_memory_allocation::Device) {
//             cudaError_t err = cudaMalloc((void**)&ptr_, size_ * sizeof(T));
//             if (err != cudaSuccess) {

//             }
//         } else if (memType == gpu_memory_allocation::Unified) {
//             cudaError_t err = cudaMallocManaged(&ptr_, size_ * sizeof(T));
//             if (err != cudaSuccess) {

//                 ptr_ = nullptr;
//             }
//         } else {
//             std::cerr << "Invalid memory type specified." << std::endl;
//             ptr_ = nullptr;
//         }
//     }

//     // Deallocate memory
//     void deallocateMemory() {
//         if (ptr_) {
//             cudaFree(ptr_);
//             ptr_ = nullptr;
//         }
//     }
// };
