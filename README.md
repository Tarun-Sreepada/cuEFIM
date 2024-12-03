# Compilation and Debugging Guide for cuEFIM

## Requirements
- **CMake 3.18+**
- **CUDA Toolkit**
- **liburing** library installed on the system.

## Steps to Compile

1. **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>

2. **Create compile environment...**
    ```bash 
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Debug
    cmake .. -DCMAKE_BUILD_TYPE=Release

3. **Compile**
    ```bash
    make -j$(nproc)

4. **Usage**
    ```bash
    ./cuEFIM -h or ./cuEFIM --help 

**Example**
```bash
./cuEFIM -f ../datasets/accidents_utility_spmf.txt -o /dev/stdout \
        -s \s -m 150000000 -p 128 -q 32 -M hash_table_shared_memory \
        -P CPU -G Device
