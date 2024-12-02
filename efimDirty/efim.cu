#include "header.cuh"
#include "parse.cu"
#include "build.cu"
#include "search.cu"
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <input file> <minutil> <output file> [method (optional)]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFileName = argv[1];
    uint32_t minutil;
    std::string outputFileName = argv[3];
    uint32_t method = 0;

    try {
        minutil = std::stoi(argv[2]);
        if (argc == 5) {
            method = std::stoi(argv[4]);
        }
    } catch (const std::invalid_argument &e) {
        std::cerr << "Error: Invalid argument provided - " << e.what() << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input file> <minutil> <output file> [method (optional)]" << std::endl;
        return EXIT_FAILURE;
    }

    auto startTime = std::chrono::steady_clock::now();

    // Retrieve GPU device properties
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    uint32_t totalSharedMem = deviceProps.sharedMemPerBlock;

    // Parse input transactions
    auto [file, item2id, id2item, twu] = parseTransactions(inputFileName);
        std::vector<std::pair<thrust::host_vector<uint32_t>, thrust::host_vector<uint32_t>>> intPatterns;


    // Build the necessary data structures
    auto [d_items, d_utils, d_indexStart, d_indexEnd, sharedMemReq,
          d_primary, d_secondaryRef, d_secondary, numSecondary] = 
          build(file, item2id, id2item, twu, minutil, totalSharedMem);

    if (method == 0) {
        if (totalSharedMem < sharedMemReq) {
            std::cerr << "Error: Not enough shared memory available." << std::endl;
            std::cerr << "Required: " << sharedMemReq << " bytes, Available: " << totalSharedMem << " bytes." << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Shared Memory: Required = " << sharedMemReq 
                  << " bytes, Available = " << totalSharedMem << " bytes." << std::endl;


        searchSM(d_items, d_utils, d_indexStart, d_indexEnd,
                 d_primary, d_secondaryRef, d_secondary, numSecondary,
                 sharedMemReq, totalSharedMem, intPatterns, minutil);
    }

    auto endTime = std::chrono::steady_clock::now();
    std::cout << "Total Execution Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() 
              << " ms" << std::endl;

    // open output file
    std::ofstream outputFile(outputFileName);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
        return EXIT_FAILURE;
    }


    //  std::vector<std::pair<thrust::host_vector<uint32_t>, thrust::host_vector<uint32_t>>> intPatterns;
    // write patterns to output file
    // for (const auto &[pattern, util] : intPatterns) {



    //     // for (const auto &item : pattern) {
    //     //     outputFile << id2item[item] << " ";
    //     // }
    //     // outputFile << "(" << util.back() << ")" << std::endl;
    // }

    for (int i = 0; i < intPatterns.size(); i++) {
        uint32_t size = i + 1;
        thrust::host_vector<uint32_t> pattern = intPatterns[i].first;
        thrust::host_vector<uint32_t> util = intPatterns[i].second;

        uint32_t number_of_patterns = pattern.size() / size;
        
        for (int j = 0; j < number_of_patterns; j++) {
            if (util[j] < minutil) {
                continue;
            }
            for (int k = 0; k < size; k++) {
                outputFile << id2item[pattern[j * size + k]] << " ";
            }
            outputFile << "(" << util[j] << ")" << std::endl;
        }
        

    }

    outputFile.close();

    return 0;
}
