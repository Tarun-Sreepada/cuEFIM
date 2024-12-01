#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#elif defined __linux__ || defined __APPLE__
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <stdio.h>
#include <numeric>
#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <tuple>

#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "vechash.cu"

#define BLOCK_SIZE 128
#define SCALING 1

#define byte sizeof(uint32_t)
#define kilo 1024
#define mega kilo *kilo
#define giga kilo *mega
