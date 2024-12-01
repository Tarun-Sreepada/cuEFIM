#include "header.cuh"
#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

// Utility to sort TWU and format as vector of pairs
std::vector<std::pair<uint32_t, uint32_t>> formatTWU(
    std::unordered_map<std::string, uint32_t> &item2id,
    std::vector<uint32_t> &twu) 
{
    std::vector<std::pair<uint32_t, uint32_t>> idTWU;
    for (const auto &[item, id] : item2id) {
        idTWU.emplace_back(id, twu[id]);
    }

    std::sort(idTWU.begin(), idTWU.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    return idTWU;
}

// Create renaming for IDs and generate a mapping
std::tuple<std::unordered_map<uint32_t, uint32_t>, std::unordered_map<uint32_t, std::string>>
createRename(const std::vector<std::pair<uint32_t, uint32_t>> &idTWU,
             const std::unordered_map<uint32_t, std::string> &id2item,
             uint32_t minutil) 
{
    std::unordered_map<uint32_t, uint32_t> id2newid;
    std::unordered_map<uint32_t, std::string> intToString;

    for (uint32_t i = 0; i < idTWU.size(); ++i) {
        if (idTWU[i].second < minutil) break;
        id2newid[idTWU[i].first] = i + 1;
        intToString[i + 1] = id2item.at(idTWU[i].first);
    }

    return {std::move(id2newid), std::move(intToString)};
}

// Merge transactions and calculate subtree utilities
std::tuple<std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, vector_hash>,
           std::unordered_map<uint32_t, uint32_t>>
merge(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &file,
      const std::unordered_map<uint32_t, uint32_t> &id2newid,
      uint32_t minutil) 
{
    std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, vector_hash> transactions;
    std::unordered_map<uint32_t, uint32_t> subtreeUtil;

    for (const auto &e : file) {
        std::vector<std::pair<uint32_t, uint32_t>> newTra;
        uint32_t utilSum = 0;

        for (const auto &[item, util] : e) {
            if (auto it = id2newid.find(item); it != id2newid.end()) {
                newTra.emplace_back(it->second, util);
                utilSum += util;
            }
        }

        std::sort(newTra.begin(), newTra.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });

        std::vector<uint32_t> items(newTra.size());
        std::vector<uint32_t> utils(newTra.size());

        uint32_t temp = 0;
        for (uint32_t i = 0; i < newTra.size(); ++i) {
            items[i] = newTra[i].first;
            utils[i] = newTra[i].second;

            subtreeUtil[items[i]] += utilSum - temp;
            temp += utils[i];
        }

        if (!newTra.empty()) {
            if (auto it = transactions.find(items); it != transactions.end()) {
                for (uint32_t i = 0; i < utils.size(); ++i) {
                    it->second[i] += utils[i];
                }
            } else {
                transactions.emplace(std::move(items), std::move(utils));
            }
        }
    }

    return {std::move(transactions), std::move(subtreeUtil)};
}

// Format transactions for GPU processing
auto formatTransactions(
    const std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, vector_hash> &transactions,
    uint32_t totalSharedMem) 
{
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> vectorTransactions;

    for (const auto &[items, utils] : transactions) {
        vectorTransactions.emplace_back(items, utils);
    }

    std::sort(vectorTransactions.begin(), vectorTransactions.end(), [](const auto &a, const auto &b) {
        return a.first.size() > b.first.size();
    });

    uint32_t sharedMemReq = 0;
    uint32_t limit = std::min(static_cast<uint32_t>(vectorTransactions.size()), static_cast<uint32_t>(128 / 16));


    for (uint32_t i = 0; i < limit; ++i) {
        sharedMemReq += vectorTransactions[i].first.size() * 2 * sizeof(uint32_t);
    }

    limit *= SCALING;

    std::vector<uint32_t> items, utils, indexStart, indexEnd;
    uint32_t prev = 0;

    for (const auto &[itemList, utilList] : vectorTransactions) {
        indexStart.push_back(prev);
        items.insert(items.end(), itemList.begin(), itemList.end());
        utils.insert(utils.end(), utilList.begin(), utilList.end());
        prev += itemList.size();
        indexEnd.push_back(prev);
    }

    return std::tuple(std::move(items), std::move(utils), std::move(indexStart), std::move(indexEnd), sharedMemReq);
}

// Transfer data to GPU
auto transferToGPU(const std::vector<uint32_t> &items,
                   const std::vector<uint32_t> &utils,
                   const std::vector<uint32_t> &indexStart,
                   const std::vector<uint32_t> &indexEnd) 
{
    thrust::device_vector<uint32_t> d_items(items.begin(), items.end());
    thrust::device_vector<uint32_t> d_utils(utils.begin(), utils.end());
    thrust::device_vector<uint32_t> d_indexStart(indexStart.begin(), indexStart.end());
    thrust::device_vector<uint32_t> d_indexEnd(indexEnd.begin(), indexEnd.end());

    return std::tuple(std::move(d_items), std::move(d_utils), std::move(d_indexStart), std::move(d_indexEnd));
}

// Process primary and secondary items
auto primarySecondary(const std::unordered_map<uint32_t, uint32_t> &subtreeUtil,
                      const std::unordered_map<uint32_t, std::string> &newIntToString,
                      uint32_t minutil) 
{
    std::vector<uint32_t> primary;
    for (const auto &[item, util] : subtreeUtil) {
        if (util >= minutil) {
            primary.push_back(item);
        }
    }

    thrust::device_vector<uint32_t> d_primary(primary.begin(), primary.end());
    thrust::device_vector<uint32_t> d_secondary(newIntToString.size() + 1, 1);
    thrust::device_vector<uint32_t> d_secondaryReference(primary.size(), 0);

    return std::tuple(std::move(d_primary), std::move(d_secondaryReference), std::move(d_secondary), 
                      primary.size(), newIntToString.size() + 1);
}

// Build all data structures
auto build(std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &file,
           std::unordered_map<std::string, uint32_t> &item2id,
           std::unordered_map<uint32_t, std::string> &id2item,
           std::vector<uint32_t> &twu,
           uint32_t minutil,
           uint32_t totalSharedMem) 
{
    auto buildStart = std::chrono::steady_clock::now();

    auto idTWU = formatTWU(item2id, twu);
    auto [id2newid, newIntToString] = createRename(idTWU, id2item, minutil);
    auto [transactions, subtreeUtil] = merge(file, id2newid, minutil);
    auto [items, utils, indexStart, indexEnd, sharedMemReq] = formatTransactions(transactions, totalSharedMem);
    auto [d_items, d_utils, d_indexStart, d_indexEnd] = transferToGPU(items, utils, indexStart, indexEnd);
    auto [d_primary, d_secondaryReference, d_secondary, numPrimary, numSecondary] = 
        primarySecondary(subtreeUtil, newIntToString, minutil);

    auto buildEnd = std::chrono::steady_clock::now();
    std::cout << "Build Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(buildEnd - buildStart).count() << "ms" << std::endl;

    return std::tuple(std::move(d_items), std::move(d_utils),
                      std::move(d_indexStart), std::move(d_indexEnd), sharedMemReq,
                      std::move(d_primary), std::move(d_secondaryReference), std::move(d_secondary), 
                      numSecondary);
}
