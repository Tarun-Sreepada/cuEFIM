#include "config.cuh"
#include "parse.cuh"
#include "mine.cuh"
#include "database.cuh"

void print_help(int argc, char *argv[])
{
    std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -f, --file <file>        Input file" << std::endl;
    std::cout << "  -o, --output <file>      Output file" << std::endl;
    std::cout << "  -s, --separator <string> Separator" << std::endl;
    std::cout << "  -m, --min_utility <int>  Minimum utility" << std::endl;
    std::cout << "  -M, --method <string>    Method to use (CPU, GPU)" << std::endl;
    std::cout << "  -h, --help               Display this information" << std::endl;
    exit(0);
}

params parse_arguments(int argc, char *argv[]) {

    params p;

    struct option long_options[] = {
        {"file", required_argument, 0, 'f'},
        {"output", required_argument, 0, 'o'},
        {"separator", required_argument, 0, 's'},
        {"min_utility", required_argument, 0, 'm'},
        {"method", no_argument, 0, 'M'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "f:o:s:m:M:h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
            case 'f':
                p.input_file = optarg;
                break;
            case 'o':
                p.output_file = optarg;
                break;
            case 's':
                p.separator = optarg;
                break;
            case 'm':
                p.min_utility = std::stoull(optarg);
                break;
            case 'M':
                p.method = optarg;
                break;
            case 'h':
                print_help(argc, argv);
                exit(0);
                break;
            default:
                std::cerr << "Invalid option" << std::endl;
                print_help(argc, argv);
                exit(1);
        }
    }

    // Validate required parameters
    if (p.input_file.empty()) {
        std::cerr << "Input file is required" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    if (p.output_file.empty()) {
        std::cerr << "Output file is required" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    if (p.separator.empty()) {
        std::cerr << "Separator is required" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    if (p.min_utility == 0) {
        std::cerr << "Minimum utility is required" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    if (p.separator != "space" && p.separator != "tab" && p.separator != "comma") {
        std::cerr << "Separator can only be space, tab, or comma" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    // Validate method parameter
    if (p.method.empty()) {
        std::cerr << "Method is required" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    if (p.method != "CPU" && p.method != "GPU") {
        std::cerr << "Method can only be CPU or GPU" << std::endl;
        print_help(argc, argv);
        exit(1);
    }

    return p;
}


int main(int argc, char *argv[]) {

    params p = parse_arguments(argc, argv);

    // print it
    std::cout << "input_file: " << p.input_file
                << "\toutput_file: " << p.output_file
                << "\tseparator: " << p.separator 
                << "\tmin_utility: " << p.min_utility
                << "\tmethod: " << p.method << std::endl;


    results r;

    r.start_time = std::chrono::high_resolution_clock::now();

    std::vector<pattern> frequent_patterns;
    auto [filtered_transactions, primary, secondary, intToStr] = parse_file(p);
    std::cout << "Time to parse file: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - r.start_time).count() << "ms" << std::endl;

    mine_patterns(p, filtered_transactions, primary, secondary, frequent_patterns, intToStr);


    r.end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Frequent Patterns: " << frequent_patterns.size() << std::endl;

    // open output file and write
    std::ofstream output_file(p.output_file);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return 1;
    }

    for (const auto &pattern : frequent_patterns) {
        for (const auto &item : pattern.items_names) {
            output_file << item << " ";
        }
        output_file << "#UTIL: " << pattern.utility << std::endl;
    }

    output_file.close();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(r.end_time - r.start_time).count();
    duration /= 1000;
    std::cout << "Execution time: " << duration << "s" << std::endl;


    return 0;

}