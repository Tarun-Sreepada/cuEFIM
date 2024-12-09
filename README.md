
---

# **Compilation and Debugging Guide for cuEFIM**

## **Requirements**
- **CMake 3.20+** (ensure compatibility with CUDA)
- **CUDA Toolkit** (CUDA 11.0 or later, as required by your GPU)
- **liburing** library (for asynchronous I/O)

---

## **Steps to Compile**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

---

### **2. Create the Build Environment**
Create a separate directory for the build files to keep the source directory clean:

#### For Debug Build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j$(nproc)

```

#### For Release Build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

---

### **3. Compile the Code**
Use `make` to build the project:

```bash
make -j$(nproc)
```

- **`-j$(nproc)`**: Utilizes all available CPU cores for faster compilation.
- The final executable will be created in the `build` directory.

---

### **4. Run the Program**
You can display the program's help menu to see all available options:

```bash
./cuEFIM -h
# or
./cuEFIM --help
```

---

## **Usage Example**

```bash
./cuEFIM -f ../datasets/accidents_utility_spmf.txt -o /dev/stdout -s \\s \
    -m 150000000 -p 128 -q 32 -M hash_table_shared_memory -P CPU -G Device
```

---

## **Program Breakdown**

### **Command-Line Arguments**
| Argument                 | Shorthand | Description                                                 | Default              |
|--------------------------|-----------|-------------------------------------------------------------|----------------------|
| `--input-file <path>`    | `-f`      | Path to the input file.                                     | *Required*          |
| `--output-file <path>`   | `-o`      | Path to the output file.                                    | `/dev/stdout`        |
| `--separator <char>`     | `-s`      | Separator character (e.g., `,`, `\s`).                     | `','`               |
| `--min-utility <value>`  | `-m`      | Minimum utility value.                                      | `0`                 |
| `--page-size <bytes>`    | `-p`      | Page size in bytes.                                         | `128 KiB`           |
| `--queue-depth <value>`  | `-q`      | Queue depth for I/O operations.                            | `512`               |
| `--read-method <method>` | `-P`      | File parsing method: `CPU` or `GPU`.                       | `CPU`               |
| `--memory <type>`        | `-G`      | GPU memory allocation: `Device`, `Unified`, or `Pinned`.    | `Device`            |
| `--method <name>`        | `-M`      | Mining method: `hash_table_shared_memory`, `no_hash_table`. | `hash_table_shared_memory` |
| `--cuda-device-id <id>`  |           | CUDA device ID to use.                                      | `0`                 |


---

## **Files in `cuEFIM`**

### **1. `main`**
- **Role**: Entry point of the program.
- **Responsibilities**:
  - Initializes the program.

---

### **2. `args`**
- **Role**: Argument parsing from the command line.
- **Responsibilities**:
  - Defines and processes command-line arguments.

---

### **3. `file`**
- **Role**: File handling based on user-specified methods.
- **Responsibilities**:
  - Opens the input file for reading.

---

### **4. `parse`**
- **Role**: Parses the file data.
- **Responsibilities**:
  - Parses data read from the file into a usable format.

---

### **5. `build`**
- **Role**: Cleans and organizes the database.
- **Responsibilities**:
  - Processes parsed data into a clean, structured format suitable for mining.

---

### **6. `convert`**
- **Role**: Converts CPU-parsed data to GPU-compatible structures.
- **Responsibilities**:
  - Prepares data structures for GPU memory (e.g., pinned memory, unified memory).

---

### **7. `mine`**
- **Role**: Performs GPU mining operations.
- **Responsibilities**:
  - Implements mining algorithms on the GPU (e.g., frequent pattern mining).

---

### **8. `output`**
- **Role**: Outputs the mining results.
- **Responsibilities**:
  - Formats and writes the results to the specified output file.

---

## **Flow of Execution**

1. **Initialization**:
   - `main` parses arguments using `args` and initializes GPU.
2. **File Reading**:
   - `file` opens and reads the input file.
3. **Parsing**:
   - `parse` converts the raw file into structured data.
4. **Database Construction**:
   - `build` organizes the parsed data into a database.
5. **GPU Preparation**:
   - `convert` transfers data to GPU-friendly formats.
6. **Mining**:
   - `mine` performs GPU mining with the selected method.
7. **Output**:
   - `output` writes results to the specified location.

---
