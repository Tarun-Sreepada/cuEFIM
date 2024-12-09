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
    // open file
    // write to file
    std::ofstream output_file(params.output_file);
    if (!output_file.is_open())
    {
        std::cerr << "Error opening output file: " << params.output_file << std::endl;
        exit(1);
    }

    // Collect frequent patterns
    // uint32_t patternCount = 0;
    // for (size_t i = 0; i < patterns.size(); ++i)
    // {
    //     for (size_t j = 0; j < patterns[i].second.size(); ++j)
    //     {
    //         if (patterns[i].second[j] < p.min_utility)
    //         {
    //             continue;
    //         }

    //         patternCount++;
    //     }
    // }
    // std::cout << "Number of patterns: " << patternCount << std::endl;

    uint32_t patternCount = 0;

    for (uint32_t i = 0; i < r.patterns.size(); i++)
    {
        for (uint32_t j = 0; j < r.patterns[i].second.size(); j++)
        {
            if (r.patterns[i].second[j] >= params.min_utility)
            {
                for (uint32_t k = 0; k < i + 1; k++)
                {
                    output_file << bf.itemID_to_item[r.patterns[i].first[(i + 1) * j + k]] << " ";
                }
                output_file << "#UTIL: " << r.patterns[i].second[j] << std::endl;
                patternCount++;
            }
        }
    }
    output_file.close();

    r.record_memory_usage("End");

    r.print_statistics();

    std::cout << "Number of patterns: " << patternCount << std::endl;


    return 0;
}