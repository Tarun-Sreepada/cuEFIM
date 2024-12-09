#include "main.cuh"
#include "include/args.cuh"
#include "include/file.hpp"
#include "include/parse.hpp"
#include "include/build.hpp"
#include "include/convert.cuh"
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

    if (params.read_method == Config::file_read_method::CPU)
    {
        // read_file_cpu(params, r);
        auto file = read_file_cpu(r, params);
        auto pf = parse_file_cpu(file, r, params);
        auto bf = build_cpu(pf, r, params);
        auto d_db = convert_to_gpu(bf, r, params);
        mine(d_db, r, params);

    }
    else
    {
        // read_file_gpu(params, r);
        // parse file
    }


    // mine


    // write to file

    r.record_memory_usage("End");

#ifdef DEBUG

    const int maxWidth = 70;
    std::string text = " Statistics ";
    int padding = (maxWidth - text.size()) / 2;

    // Create the left and right padding
    std::string leftPadding(padding, '-');
    std::string rightPadding(maxWidth - text.size() - leftPadding.size(), '-');

    // Print the formatted output
    std::cout << leftPadding << text << rightPadding << std::endl;
    std::cout << std::setw(15) << "Label"
              << std::setw(20) << "Time (s.ms)"
              << std::setw(15) << "RSS (MB)"
              << std::setw(20) << "CUDA Memory (MB)" << std::endl;
    std::cout << std::string(maxWidth, '-') << std::endl;

    // Data
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    for (const auto &[label, data] : r.memory_usage)
    {
        auto [timestamp, rss, cuda_mem] = data;

        // If the label is "Start", record the start time
        if (label == "Start")
        {
            start_time = timestamp;
            std::cout << std::setw(15) << label
                      << std::setw(20) << "0.000" // Start time is 0
                      << std::setw(15) << rss / 1024 / 1024
                      << std::setw(20) << cuda_mem / 1024 / 1024
                      << std::endl;
            continue;
        }

        // Calculate time difference from the start time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - start_time);
        double seconds_ms = duration.count() / 1000.0;

        // Print the data
        std::cout << std::setw(15) << label
                  << std::setw(20) << std::fixed << std::setprecision(3) << seconds_ms
                  << std::setw(15) << rss / 1024 / 1024
                  << std::setw(20) << cuda_mem / 1024 / 1024
                  << std::endl;
    }

#endif

    return 0;
}