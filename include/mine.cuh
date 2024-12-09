#pragma once
#include "../main.cuh"
#include "args.cuh"
#include "structure.cuh"
#include "memory.cuh"

void mine(CudaMemory<d_database> &db, results &r, Config::Params &p);
