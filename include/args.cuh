#pragma once
#include "../main.cuh"
#include <cuda_runtime.h>

namespace Config {

inline void print_gpu_stats()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
        return;
    }

    if (device_count == 0) {
        std::cout << "No CUDA-capable GPUs detected.\n";
        return;
    }

    std::cout << "Detected " << device_count << " CUDA-capable GPU(s):\n";

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nGPU " << i << ": " << prop.name << "\n"
                  << "  Compute Capability: " << prop.major << "." << prop.minor << "\n"
                  << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n"
                  << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB\n"
                  << "  Registers per Block: " << prop.regsPerBlock << "\n"
                  << "  Warp Size: " << prop.warpSize << " threads\n"
                  << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Max Threads Dimensions: (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n"
                  << "  Max Grid Size: (" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n"
                  << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n"
                  << "  Memory Bus Width: " << prop.memoryBusWidth << "-bit\n"
                  << "  Peak Memory Bandwidth: "
                  << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << " GB/s\n";
    }
}

enum class gpu_memory_allocation
{
    Device,  // Can only be accessed by the GPU
    Unified, // Can be accessed by both CPU and GPU
    Pinned,  // Host memory that is pinned
};

enum class mine_method
{
    no_hash_table,
    no_hash_table_shared_memory,
    hash_table,
    hash_table_shared_memory,
};

enum class file_read_method
{
    CPU,
    GPU
};

struct Params
{
    std::string input_file;
    std::string output_file = "/dev/stdout"; // Default to /dev/stdout
    std::string separator;

    char separator_char = ',';

    uint32_t min_utility = 0;

    size_t page_size = 128 * 1024; // KIBIBYTE assumed to be 1024
    size_t queue_depth = 512;

    file_read_method read_method = file_read_method::CPU;
    gpu_memory_allocation GPU_memory_allocation = gpu_memory_allocation::Device;
    mine_method method = mine_method::hash_table_shared_memory;

    int cuda_device_id = 0; // Default CUDA device ID
};

void set_device(int device_id)
{
    cudaDeviceReset();
    cudaError_t err = cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        std::cerr << "Cannot set CUDA device: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}

inline void print_help(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --input-file <path>       Path to the input file\n"
              << "  --output-file <path>      Path to the output file (default: /dev/stdout)\n"
              << "  --separator <char>        Separator character (default: ',')\n"
              << "  --min-utility <value>     Minimum utility value (default: 0)\n"
              << "  --page-size <bytes>       Page size in bytes (default: 128 KiB)\n"
              << "  --queue-depth <value>     Queue depth (default: 512)\n"
              << "  --read-method <CPU|GPU>   File parsing method (default: CPU)\n"
              << "  --memory <Device|Unified|Pinned> GPU memory allocation (default: Device)\n"
              << "  --method <name>           Mining method (default: hash_table_shared_memory)\n"
              << "  --cuda-device-id <id>     CUDA device ID (default: 0)\n";
}

inline Params parse_arguments(int argc, char* argv[])
{
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input-file" && i + 1 < argc) {
            p.input_file = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            p.output_file = argv[++i];
        } else if (arg == "--separator" && i + 1 < argc) {
            p.separator = argv[++i];
            if (!p.separator.empty()) {
                p.separator_char = p.separator[0];
            }
        } else if (arg == "--min-utility" && i + 1 < argc) {
            p.min_utility = std::stoul(argv[++i]);
        } else if (arg == "--page-size" && i + 1 < argc) {
            p.page_size = std::stoull(argv[++i]);
        } else if (arg == "--queue-depth" && i + 1 < argc) {
            p.queue_depth = std::stoul(argv[++i]);
        } else if (arg == "--read-method" && i + 1 < argc) {
            std::string method = argv[++i];
            if (method == "CPU") {
                p.read_method = file_read_method::CPU;
            } else if (method == "GPU") {
                p.read_method = file_read_method::GPU;
            }
        } else if (arg == "--memory" && i + 1 < argc) {
            std::string mem = argv[++i];
            if (mem == "Device") {
                p.GPU_memory_allocation = gpu_memory_allocation::Device;
            } else if (mem == "Unified") {
                p.GPU_memory_allocation = gpu_memory_allocation::Unified;
            } else if (mem == "Pinned") {
                p.GPU_memory_allocation = gpu_memory_allocation::Pinned;
            }
        } else if (arg == "--method" && i + 1 < argc) {
            std::string method = argv[++i];
            if (method == "no_hash_table") {
                p.method = mine_method::no_hash_table;
            } else if (method == "no_hash_table_shared_memory") {
                p.method = mine_method::no_hash_table_shared_memory;
            } else if (method == "hash_table") {
                p.method = mine_method::hash_table;
            } else if (method == "hash_table_shared_memory") {
                p.method = mine_method::hash_table_shared_memory;
            }
        } else if (arg == "--cuda-device-id" && i + 1 < argc) {
            p.cuda_device_id = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_help(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            exit(1);
        }
    }

    return p;
}

inline void print_arguments(const Params& p)
{
    std::cout << "\n\tInput File: " << p.input_file << "\n"
              << "\tOutput File: " << p.output_file << "\n"
              << "\tSeparator: " << p.separator_char << "\n"
              << "\tMinimum Utility: " << p.min_utility << "\n"
              << "\tPage Size: " << p.page_size << "\n"
              << "\tQueue Depth: " << p.queue_depth << "\n"
              << "\tRead Method: " << (p.read_method == file_read_method::CPU ? "CPU" : "GPU") << "\n"
              << "\tGPU Memory Allocation: "
              << (p.GPU_memory_allocation == gpu_memory_allocation::Device
                      ? "Device"
                      : p.GPU_memory_allocation == gpu_memory_allocation::Unified ? "Unified" : "Pinned")
              << "\n"
              << "\tMining Method: "
              << (p.method == mine_method::no_hash_table
                      ? "no_hash_table"
                      : p.method == mine_method::no_hash_table_shared_memory
                            ? "no_hash_table_shared_memory"
                            : p.method == mine_method::hash_table
                                  ? "hash_table"
                                  : "hash_table_shared_memory")
              << "\n"
              << "\tCUDA Device ID: " << p.cuda_device_id << "\n";
}

} // namespace Config
