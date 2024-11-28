#include "main.hpp"
#include "parse.hpp"

void print_help(int argc, char *argv[])
{
    std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -f, --file <file>        Input file" << std::endl;
    std::cout << "  -o, --output <file>      Output file" << std::endl;
    std::cout << "  -s, --separator <string> Separator" << std::endl;
    std::cout << "  -m, --min_utility <int>  Minimum utility" << std::endl;
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
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "f:o:s:m:h", long_options, &option_index);

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
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -f, --file <file>        Input file" << std::endl;
                std::cout << "  -o, --output <file>      Output file" << std::endl;
                std::cout << "  -s, --separator <string> Separator" << std::endl;
                std::cout << "  -m, --min_utility <int>  Minimum utility" << std::endl;
                std::cout << "  -h, --help               Display this information" << std::endl;
                exit(0);
                break;
            default:
                std::cerr << "Invalid option" << std::endl;
                exit(1);
        }
    }


    // if input_file is empty, print help
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

    // seperator can only be space or tab or comma but we can only type space tab or comma
    if (p.separator != "space" && p.separator != "tab" && p.separator != "comma") {
        std::cerr << "Separator can only be space, tab or comma" << std::endl;
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

    std::vector<std::pair<std::vector<std::string>, uint64_t>> patterns;

    auto [strToInt, intToStr, filtered_transactions, sorted_twu, primary, secondary] = parse_file(p);

    for (auto &item : primary) {
        std::vector<std::string> pattern;
        pattern.push_back(intToStr[item]);
        patterns.push_back({pattern, i
    }

}