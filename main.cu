#include "main.cuh"
#include "include/args.cuh"
#include "include/file.hpp"
#include "include/parse.hpp"
#include "include/build.hpp"
#include "include/mine.cuh"



int main(int argc, char *argv[])
{

#ifdef DEBUG
    std::cout << "DEBUG MODE" << std::endl;
#endif

    auto params = Config::parse_arguments(argc, argv);
    Config::set_device(params.cuda_device_id, params);
    Config::print_arguments(params);

    results r;
    r.record_memory_usage("Start");

    auto file = read_file_cpu(r, params);
    auto pf = parse_file_cpu(file, r, params);
    auto bf = build_cpu(pf, r, params);
    mine(bf, r, params);


    // mine


    // write to file

    r.record_memory_usage("End");

    r.print_statistics();

    return 0;
}