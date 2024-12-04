#include "include/parse.cuh"
#include "include/memory.cuh"
#include "include/structure.cuh"
#include "include/file.hpp"


int main(int argc, char *argv[]) {

    params p = parse_arguments(argc, argv);
    print_arguments(p);

    results r;
    r.start_time = std::chrono::high_resolution_clock::now();
    r.gpu_memory_consumption_before_starting = used_gpu_memory();

    read_file(r,p);

    std::cout << "Time to read file: " << std::chrono::duration_cast<std::chrono::milliseconds>(r.file_read_time - r.start_time).count() << "ms" << std::endl;
    std::cout << "Time to parse file: " << std::chrono::duration_cast<std::chrono::milliseconds>(r.parse_time - r.file_read_time).count() << "ms" << std::endl;

    // std::vector<pattern> frequent_patterns;
    // auto [filtered_transactions, primary, secondary, intToStr] = parse_file(p);
    // std::cout << "Time to parse file: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - r.start_time).count() << "ms" << std::endl;

    // mine_patterns(p, filtered_transactions, primary, secondary, frequent_patterns, intToStr);


    // r.end_time = std::chrono::high_resolution_clock::now();
    // std::cout << "Frequent Patterns: " << frequent_patterns.size() << std::endl;

    // // open output file and write
    // std::ofstream output_file(p.output_file);
    // if (!output_file.is_open()) {
    //     std::cerr << "Error opening output file" << std::endl;
    //     return 1;
    // }

    // for (const auto &pattern : frequent_patterns) {
    //     for (const auto &item : pattern.items_name


    return 0;

}