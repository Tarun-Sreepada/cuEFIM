cmake_minimum_required(VERSION 3.20)
project(cuEFIM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "75;80")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include)
find_library(LIBURING_LIB NAMES uring REQUIRED)
find_path(LIBURING_INCLUDE_DIR NAMES liburing.h REQUIRED)
include_directories(${LIBURING_INCLUDE_DIR})

file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_CU_FILES src/*.cu)

# Define the CUDA executable
add_executable(cuEFIM ${SRC_CPP_FILES} ${SRC_CU_FILES} main.cu)

# Set properties to specify CUDA-specific flags
set_target_properties(cuEFIM PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# Set custom flags for Debug and Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -g")
    target_compile_definitions(cuEFIM PRIVATE DEBUG) # Define DEBUG only in Debug mode
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Link necessary libraries
target_link_libraries(cuEFIM PRIVATE ${LIBURING_LIB} cuda cudart)


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")


message(STATUS "--------------------------------------------------------------------")
message(STATUS "cuEFIM:")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "To compile, run:")
message(STATUS "  make -j$(nproc)")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "Usage Instructions for cuEFIM:")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "To display help:")
message(STATUS "  ./cuEFIM -h")
message(STATUS "  ./cuEFIM --help")
message(STATUS "")
message(STATUS "To run cuEFIM:")
message(STATUS " \n ./cuEFIM --input-file /datasets/accidents_utility_spmf.txt --output-file /dev/stdout --separator \\s --min-utility 50000000 --page-size 128 --queue-depth 32 --read-method CPU --memory Device --method hash_table_shared_memory")
message(STATUS "--------------------------------------------------------------------")

