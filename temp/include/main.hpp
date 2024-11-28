#pragma once
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <getopt.h>


struct params
{
    std::string input_file;
    std::string output_file;
    std::string separator;

    uint64_t min_utility = 0;
};

params parse_arguments(int argc, char *argv[]);