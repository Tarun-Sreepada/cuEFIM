#include "config.cuh"
#include "parse.cuh"
#include "mine.cuh"

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
        {"method", required_argument, 0, 'M'},
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
                << "\tseparator: '" << p.separator 
                << "\tmin_utility: " << p.min_utility << std::endl;


    results r;

    r.start_time = std::chrono::high_resolution_clock::now();

    auto [strToInt, intToStr, filtered_transactions, sorted_twu, subtree_util, secondary_util] = parse_file(p);

    // for item in subtree util if less than min_utility then remove it
    for (auto it = subtree_util.begin(); it != subtree_util.end(); ) {
        if (it->second < p.min_utility) {
            it = subtree_util.erase(it);
        } else {
            ++it;
        }
    }

    // for item in secondary util if less than min_utility then remove it
    for (auto it = secondary_util.begin(); it != secondary_util.end(); ) {
        if (it->second < p.min_utility) {
            it = secondary_util.erase(it);
        } else {
            ++it;
        }
    }

    std::cout << "Subtree util: " << subtree_util.size() << std::endl;
    std::cout << "Secondary util: " << secondary_util.size() << std::endl;

    // r.frequentItemsets = generate_frequent_itemsets(filtered_transactions, subtree_util, secondary_util, p.min_utility);
    if (p.method == "CPU") {
        r.frequentItemsets = generate_frequent_itemsets_cpu(filtered_transactions, subtree_util, secondary_util, p.min_utility);
    } else {
        r.frequentItemsets = generate_frequent_itemsets_gpu(filtered_transactions, subtree_util, secondary_util, p.min_utility);
    }

    r.end_time = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(r.end_time - r.start_time).count();
    duration /= 1000;
    std::cout << "Execution time: " << duration << "s" << std::endl;
    std::cout << "Frequent itemsets: " << r.frequentItemsets.size() << std::endl;


    return 0;

}