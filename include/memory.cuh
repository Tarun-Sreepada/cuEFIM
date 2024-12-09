#pragma once
#include "../main.cuh"
#include "args.cuh"


template <typename T>
class CudaMemory
{
public:
    // Constructor with optional allocation
    CudaMemory(size_t size = 0, Config::gpu_memory_allocation memType = Config::gpu_memory_allocation::Device)
        : size_(size), ptr_(nullptr)
    {
        if (size > 0)
        {
            allocateMemory(memType);
        }
    }

    // Constructor that takes a std::vector and allocates memory
    CudaMemory(const std::vector<T>& vec, Config::gpu_memory_allocation memType = Config::gpu_memory_allocation::Device)
        : size_(vec.size()), ptr_(nullptr)
    {
        if (size_ > 0)
        {
            allocateMemory(memType);
            cudaMemcpy(ptr_, vec.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    // Destructor
    ~CudaMemory()
    {
        deallocateMemory();
    }

    // Copy constructor
    CudaMemory(const CudaMemory &other)
        : size_(other.size_), ptr_(nullptr)
    {
        if (size_ > 0)
        {
            allocateMemory(Config::gpu_memory_allocation::Device);
            cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    // Copy assignment operator
    CudaMemory &operator=(const CudaMemory &other)
    {
        if (this != &other)
        {
            deallocateMemory();
            size_ = other.size_;
            if (size_ > 0)
            {
                allocateMemory(Config::gpu_memory_allocation::Device);
                cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    // Move constructor
    CudaMemory(CudaMemory &&other) noexcept
        : size_(other.size_), ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment operator
    CudaMemory &operator=(CudaMemory &&other) noexcept
    {
        if (this != &other)
        {
            deallocateMemory();
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Accessor for the GPU pointer
    T* ptr() const __host__ __device__ { return ptr_; }

    // Retrieve data from the GPU to the host
    std::vector<T> get() const
    {
        if (!ptr_)
        {
            return {};
        }
        std::vector<T> h_vec(size_);
        cudaMemcpy(h_vec.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        return h_vec;
    }

    // Copy data from std::vector to the current CudaMemory instance
    void copy_from_host(const std::vector<T>& vec)
    {
        size_ = vec.size();
        allocateMemory(Config::gpu_memory_allocation::Device);
        cudaMemcpy(ptr_, vec.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

private:
    size_t size_;
    T *ptr_;

    // Allocate memory
    void allocateMemory(Config::gpu_memory_allocation memType)
    {
        cudaError_t err;
        switch (memType)
        {
        case Config::gpu_memory_allocation::Device:
            err = cudaMalloc((void **)&ptr_, size_ * sizeof(T));
            break;
        case Config::gpu_memory_allocation::Unified:
            err = cudaMallocManaged(&ptr_, size_ * sizeof(T));
            break;
        case Config::gpu_memory_allocation::Pinned:
            err = cudaMallocHost((void **)&ptr_, size_ * sizeof(T));
            break;
        default:
            std::cerr << "Invalid memory type specified." << std::endl;
            ptr_ = nullptr;
            return;
        }

        if (err != cudaSuccess)
        {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
            ptr_ = nullptr;
        }
    }

    // Deallocate memory
    void deallocateMemory()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
};
