#include "header.cuh"

#include <set>


__device__ int32_t binarySearch(uint32_t *arr, uint32_t l, uint32_t r, uint32_t x)
{

    if (arr[l] == x)
        return l;
    else if (arr[r] == x)
        return r;
    else if (arr[l] < x)
        return -1;
    else if (arr[r] > x)
        return -1;

    while (l <= r) {
        int mid = (l + r) / 2;
        if (arr[mid] > x)
            l = mid + 1;
        else if (arr[mid] < x)
            r = mid - 1;
        else
            return mid;
    }

    return -1;
}

__device__ int32_t linearSearch(uint32_t *arr, uint32_t l, uint32_t r, uint32_t x)
{
    
    for (uint32_t i = l; i < r; i++)
    {
        if (arr[i] == x)
            return i;
        if (arr[i] < x)
            return -1;
    }

    return -1;
}

extern __shared__ uint32_t shared[];


__global__ void searchGPUsmScaling(uint32_t *items, uint32_t *utils, uint32_t *indexStart, uint32_t *indexEnd, uint32_t *hits, uint32_t numTransactions,
                            uint32_t *candidates, uint32_t candidateSize, uint32_t numCandidates,
                            uint32_t *candidateCost, uint32_t *candidateLocalUtil, uint32_t *candidateSubtreeUtil,
                            uint32_t *secondaryReference, uint32_t *secondaries, uint32_t numSecondaries, uint32_t scaling)
{

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t line = tid / scaling;
    uint32_t candOffset = tid % scaling;

    if (line >= numTransactions)
        return;

    // load items and utils
    uint32_t start = indexStart[line];
    uint32_t end = indexEnd[line];
    uint32_t size = end - start;
    uint32_t hit = 0;

    // Calculate the starting index in shared memory for the current transaction
    uint32_t sharedLoc = line % (BLOCK_SIZE / scaling); // lines % number of lines per block


    uint32_t sharedStart = SCALING * 3 * sharedLoc;
    for (uint32_t i = line - sharedLoc; i < line; i++)
    {
        sharedStart += 2 * (indexEnd[i] - indexStart[i]);
    }

    uint32_t sharedItemStart = sharedStart; // + candOffset * 2 * size;
    uint32_t sharedUtilStart = sharedStart + size; // + candOffset * 2 * size;

    // Copy items and utils to shared memory

    for (uint32_t i = start + candOffset; i < end; i += scaling)
    {
        shared[sharedItemStart + i - start] = items[i];
    }

    for (uint32_t i = start + candOffset; i < end; i += scaling)
    {
        shared[sharedUtilStart + i - start] = utils[i];
    }

    uint32_t spBME = sharedUtilStart + size + 3 * candOffset;
    shared[spBME] = shared[sharedItemStart];
    shared[spBME + 1] = shared[(sharedItemStart + size - 1) / 2];
    shared[spBME + 2] = shared[sharedItemStart + size - 1];

    __syncthreads();


    // Iterate over the candidates
    for (uint32_t i = candOffset; i < numCandidates; i += scaling)
    {
        
        uint32_t found = 0;
        uint32_t foundCost = 0;
        uint32_t foundLoc = 0;

        for (uint32_t j = 0; j < candidateSize; j++)
        {
            uint32_t currCand = candidates[i * candidateSize + j];

            int32_t loc = 0;
            if (shared[spBME] == currCand)
                loc = sharedItemStart;
            else if (shared[spBME + 2] == currCand)
                loc = sharedItemStart + size - 1;
            else if (shared[spBME] < currCand)
                loc = -1;
            else if (shared[spBME + 2] > currCand) 
                loc = -1;
            else
                loc = binarySearch(shared, sharedItemStart + 1, sharedUtilStart - 2, currCand);

            if (loc != -1)
            {
                found++;
                foundCost += shared[loc + size];
                foundLoc = loc;
            }
            else
            {
                break;
            }
        }

        if (found != candidateSize)
            continue;


        // Update the candidate cost
        uint32_t ret = atomicAdd(&candidateCost[i], foundCost);

        hit = 1;

        // uint32_t oldcost = foundCost;

        // Update candidateLocalUtil and candidateSubtreeUtil
        for (uint32_t j = foundLoc + 1; j < sharedUtilStart; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + shared[j]])
            {
                foundCost += shared[j + size];
            }
        }

        uint32_t temp = 0;
        for (uint32_t j = foundLoc + 1; j < sharedUtilStart; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + shared[j]])
            {
                uint32_t ret1 = atomicAdd(&candidateLocalUtil[i * numSecondaries + shared[j]], foundCost);
                uint32_t ret2 = atomicAdd(&candidateSubtreeUtil[i * numSecondaries + shared[j]], foundCost - temp);

                temp += shared[j + size];
            }
        }
    }

    if (hit)
    {
        hits[line] = 1;
    }

}


__global__ void searchGPU(uint32_t *items, uint32_t *utils, uint32_t *indexStart, uint32_t *indexEnd, uint32_t *hits, uint32_t numTransactions,
                            uint32_t *candidates, uint32_t candidateSize, uint32_t numCandidates,
                            uint32_t *candidateCost, uint32_t *candidateLocalUtil, uint32_t *candidateSubtreeUtil,
                            uint32_t *secondaryReference, uint32_t *secondaries, uint32_t numSecondaries, uint32_t scaling)
{

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t line = tid / scaling;
    uint32_t candOffset = tid % scaling;

    if (line >= numTransactions)
        return;

    // load items and utils
    uint32_t start = indexStart[line];
    uint32_t end = indexEnd[line];

    uint32_t hit = 0;

    // Iterate over the candidates
    for (uint32_t i = candOffset; i < numCandidates; i += scaling)
    {
        
        uint32_t found = 0;
        uint32_t foundCost = 0;
        uint32_t foundLoc = 0;

        for (uint32_t j = 0; j < candidateSize; j++)
        {
            uint32_t currCand = candidates[i * candidateSize + j];
            int32_t loc = binarySearch(items, start, end - 1, currCand);
            // int32_t loc = linearSearch(items, start, end, currCand);

            if (loc != -1)
            {
                found++;
                foundCost += utils[loc];
                foundLoc = loc;
            }
            else
            {
                break;
            }
        }

        if (found != candidateSize)
            continue;

        // Update the candidate cost
        uint32_t ret = atomicAdd(&candidateCost[i], foundCost);
        hit = 1;

        for (uint32_t j = foundLoc + 1; j < end; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + items[j]])
            {
                foundCost += utils[j];
            }
        }

        uint32_t temp = 0;

        for (uint32_t j = foundLoc + 1; j < end; j++)
        {
            if (secondaries[secondaryReference[i] * numSecondaries + items[j]])
            {
                uint32_t ret1 = atomicAdd(&candidateLocalUtil[i * numSecondaries + items[j]], foundCost);
                uint32_t ret2 = atomicAdd(&candidateSubtreeUtil[i * numSecondaries + items[j]], foundCost - temp);
                temp += utils[j];
            }
        }
    }

    if (hit)
    {
        hits[line] = 1;
    }

}


__global__ void cleanSubtreeLocal(uint32_t numCands, uint32_t *numNewCand, uint32_t *candSubtreeUtil, uint32_t *candLocalUtil, uint32_t numSecondaries, uint32_t minUtil)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numCands)
        return;

    for (uint32_t i = tid * numSecondaries; i < (tid + 1) * numSecondaries; i++)
    {
        if (candSubtreeUtil[i] >= minUtil)
        {
            candSubtreeUtil[i] = i - tid * numSecondaries;
            numNewCand[tid + 1]++;
        }
        else
        {
            candSubtreeUtil[i] = 0;
        }
        if (candLocalUtil[i] >= minUtil)
        {
            candLocalUtil[i] = i - tid * numSecondaries;
        }
        else
        {
            candLocalUtil[i] = 0;
        }
    }
    return;
}

__global__ void createNewCands(uint32_t *cands, uint32_t *candSubtreeUtil, uint32_t numCands, uint32_t *newCands,
                               uint32_t *newSecondaryRefs, uint32_t numSecondaries, uint32_t size, uint32_t *locs)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numCands)
        return;

    if (locs[tid] == locs[tid + 1])
        return;

    uint32_t counter = size * locs[tid];
    uint32_t refStart = locs[tid];
    for (uint32_t i = tid * numSecondaries; i < (tid + 1) * numSecondaries; i++)
    {
        if (candSubtreeUtil[i])
        {
            for (uint32_t j = tid * (size - 1); j < (tid + 1) * (size - 1); j++)
            {
                newCands[counter] = cands[j];
                counter++;
            }
            newCands[counter] = candSubtreeUtil[i];
            counter++;
            newSecondaryRefs[refStart] = tid;
            refStart++;
        }
    }
}

__global__ void writeNewIndex(uint32_t *start, uint32_t *end, uint32_t *act, uint32_t *loc,
                              uint32_t *newStart, uint32_t *newEnd,
                              uint32_t numTrans)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= numTrans)
        return;

    if (act[tid])
    {
        uint32_t locMinusOne = loc[tid] - 1;
        newStart[locMinusOne] = start[tid];
        newEnd[locMinusOne] = end[tid];
    }
}

struct is_greater
{
  __host__ __device__
  bool operator()(int &x, int &y)
  {
    return x > y;
  }
};

void printVector(thrust::host_vector<uint32_t> vec)
{
    for (uint32_t i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void printVector(thrust::host_vector<uint32_t> vec, uint32_t candSize)
{
    for (uint32_t i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << " ";
        if ((i + 1) % candSize == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}


int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

void searchSM(thrust::device_vector<uint32_t> items, thrust::device_vector<uint32_t> utilities,
              thrust::device_vector<uint32_t> indexStart, thrust::device_vector<uint32_t> indexEnd,
              thrust::device_vector<uint32_t> cands, thrust::device_vector<uint32_t> secondaryRef, thrust::device_vector<uint32_t> secondary,
              uint32_t numSecondary,
              uint32_t sharedMemReq, uint32_t totalSharedMem,
              std::vector<std::pair<thrust::host_vector<uint32_t>, thrust::host_vector<uint32_t>>> &intPatterns, uint32_t minutil)
{

    uint32_t candSize = 1;
    uint32_t numCands = cands.size();
    uint32_t numberOfLines = indexStart.size();
    uint32_t patternCount = 0;

    sharedMemReq += SCALING * 4 * (BLOCK_SIZE / SCALING);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t coreCount = getSPcores(prop);
    // std::cout << "Number of cores: " << coreCount << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    while (numCands)
    {
        std::cout << "Number of candidates: " << numCands << std::endl;
        // std::cout << "Number of lines: " << numberOfLines << std::endl;
        // std::cout << std::endl;

        thrust::device_vector<uint32_t> subtreeUtils(numCands * numSecondary);
        thrust::device_vector<uint32_t> localUtils(numCands * numSecondary);
        thrust::device_vector<uint32_t> costs(numCands);
        thrust::device_vector<uint32_t> hits(numberOfLines);

        thrust::fill(subtreeUtils.begin(), subtreeUtils.end(), 0);
        thrust::fill(localUtils.begin(), localUtils.end(), 0);
        thrust::fill(costs.begin(), costs.end(), 0);
        thrust::fill(hits.begin(), hits.end(), 0);

        // print number lines divided by core count in percent
        float percent = (float)numberOfLines / (float)coreCount;
    
        uint32_t blocks =  SCALING * (numberOfLines + BLOCK_SIZE) / BLOCK_SIZE;

        // searchGPUsmScaling<<<blocks, BLOCK_SIZE, sharedMemReq>>>(thrust::raw_pointer_cast(items.data()), thrust::raw_pointer_cast(utilities.data()),
        //                                           thrust::raw_pointer_cast(indexStart.data()), thrust::raw_pointer_cast(indexEnd.data()),
        //                                           thrust::raw_pointer_cast(hits.data()), numberOfLines,
        //                                           thrust::raw_pointer_cast(cands.data()), candSize, numCands,
        //                                           thrust::raw_pointer_cast(costs.data()), thrust::raw_pointer_cast(localUtils.data()), thrust::raw_pointer_cast(subtreeUtils.data()),
        //                                           thrust::raw_pointer_cast(secondaryRef.data()), thrust::raw_pointer_cast(secondary.data()), numSecondary, SCALING);



        searchGPU<<<blocks, BLOCK_SIZE, sharedMemReq>>>(thrust::raw_pointer_cast(items.data()), thrust::raw_pointer_cast(utilities.data()),
                                                  thrust::raw_pointer_cast(indexStart.data()), thrust::raw_pointer_cast(indexEnd.data()),
                                                  thrust::raw_pointer_cast(hits.data()), numberOfLines,
                                                  thrust::raw_pointer_cast(cands.data()), candSize, numCands,
                                                  thrust::raw_pointer_cast(costs.data()), thrust::raw_pointer_cast(localUtils.data()), thrust::raw_pointer_cast(subtreeUtils.data()),
                                                  thrust::raw_pointer_cast(secondaryRef.data()), thrust::raw_pointer_cast(secondary.data()), numSecondary, SCALING);


        cudaDeviceSynchronize();


        patternCount += thrust::count_if(thrust::device, costs.begin(), costs.end(), thrust::placeholders::_1 >= minutil);
        // std::cout << "Pattern count: " << patternCount << std::endl;

        thrust::host_vector<uint32_t> h_cands(cands.size());
        thrust::host_vector<uint32_t> h_costs(costs.size());
        thrust::copy(cands.begin(), cands.end(), h_cands.begin());
        thrust::copy(costs.begin(), costs.end(), h_costs.begin());

        std::set<std::vector<uint32_t>> patterns;

        intPatterns.push_back(std::make_pair(h_cands, h_costs));

        candSize++;
        thrust::device_vector<uint32_t> hitcopy(hits.size());
        thrust::copy(hits.begin(), hits.end(), hitcopy.begin());
        uint32_t hitcount = thrust::reduce(hits.begin(), hits.end());
        thrust::inclusive_scan(hits.begin(), hits.end(), hits.begin());

        thrust::device_vector<uint32_t> newIndexesStart(hitcount);
        thrust::device_vector<uint32_t> newIndexesEnd(hitcount);

        writeNewIndex<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(indexStart.data()), thrust::raw_pointer_cast(indexEnd.data()), thrust::raw_pointer_cast(hitcopy.data()), thrust::raw_pointer_cast(hits.data()),
                                              thrust::raw_pointer_cast(newIndexesStart.data()), thrust::raw_pointer_cast(newIndexesEnd.data()),
                                              numberOfLines);

        cudaDeviceSynchronize();
        // std::cout << "write finished" << std::endl;


        thrust::device_vector<uint32_t> numNewCandsPerCand(numCands + 1);
        thrust::fill(numNewCandsPerCand.begin(), numNewCandsPerCand.end(), 0);

        blocks = (numCands + BLOCK_SIZE) / BLOCK_SIZE;

        cleanSubtreeLocal<<<blocks, BLOCK_SIZE>>>(numCands, thrust::raw_pointer_cast(numNewCandsPerCand.data()), thrust::raw_pointer_cast(subtreeUtils.data()), thrust::raw_pointer_cast(localUtils.data()), numSecondary, minutil);
        uint32_t totalNumNewCands = thrust::reduce(thrust::device, numNewCandsPerCand.begin(), numNewCandsPerCand.end());

        thrust::inclusive_scan(numNewCandsPerCand.begin(), numNewCandsPerCand.end(), numNewCandsPerCand.begin());

        thrust::device_vector<uint32_t> newCands(totalNumNewCands * candSize);
        thrust::device_vector<uint32_t> newSecondaryRefs(totalNumNewCands);

        createNewCands<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(cands.data()), thrust::raw_pointer_cast(subtreeUtils.data()), numCands,
                                               thrust::raw_pointer_cast(newCands.data()), thrust::raw_pointer_cast(newSecondaryRefs.data()), numSecondary, candSize, thrust::raw_pointer_cast(numNewCandsPerCand.data()));

        cudaDeviceSynchronize();

        cands = newCands;
        secondary = localUtils;
        secondaryRef = newSecondaryRefs;
        numCands = totalNumNewCands;

        // numCands = 0;

        indexStart = newIndexesStart;
        indexEnd = newIndexesEnd;
        numberOfLines = hitcount;

    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << "ms" << std::endl;

    std::cout << "Number of patterns: " << patternCount << std::endl;

}
