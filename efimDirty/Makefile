# Set the compiler and compiler flags
NVCC = nvcc
NVCCFLAGS = -arch=sm_86 -O3 -std=c++20

# Define the object files
OBJ = efim.o parse.o search.o vechash.o build.o

# Define the target executable
TARGET = efim

# Build the target executable
$(TARGET):
	$(NVCC) $(NVCCFLAGS) efim.cu -o efim