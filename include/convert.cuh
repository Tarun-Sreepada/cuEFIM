#pragma once
#include "../main.cuh"
#include "build.hpp"
#include "memory.cuh"
#include "structure.hpp"
#include "structure.cuh"


CudaMemory<d_database> convert_to_gpu(build_file &bf, results &r, Config::Params &p);