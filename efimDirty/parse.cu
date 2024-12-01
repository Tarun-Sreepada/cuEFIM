#include "header.cuh"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

// Parse a single line of transaction data
auto parseTransactionOneLine(
    std::string line,
    std::unordered_map<std::string, uint32_t> &item2id,
    std::unordered_map<uint32_t, std::string> &id2item,
    std::vector<uint32_t> &twu) 
{
    std::vector<std::pair<uint32_t, uint32_t>> transaction;
    uint32_t transactionUtility = 0;

    try {
        std::size_t i = 0, j = 0;
        std::vector<std::pair<uint32_t, uint32_t>> buffer;

        // Parse items in the transaction
        while (i < line.size()) {
            for (std::size_t k = i; k < line.size(); ++k) {
                if (line[k] == ' ' || line[k] == '\t' || line[k] == ':') {
                    j = k - i;
                    break;
                }
            }
            std::string item = line.substr(i, j);
            auto it = item2id.find(item);

            uint32_t id;
            if (it != item2id.end()) {
                id = it->second;
            } else {
                id = static_cast<uint32_t>(item2id.size());
                item2id[item] = id;
                id2item[id] = item;
                twu.push_back(0);
            }

            buffer.emplace_back(id, 0);
            i += j + 1;

            if (line[i - 1] == ':') break; // Exit if we've reached the utility section
        }

        // Parse transaction utility
        transaction.reserve(buffer.size());
        for (const auto &pair : buffer) transaction.push_back(pair);

        transactionUtility = static_cast<uint32_t>(std::stol(line.data() + i, &j));
        i += j + 1;

        if (line[i - 1] != ':') throw std::runtime_error("Failed to parse transaction utility");

        // Parse utilities for each item
        for (auto &[item, utility] : transaction) {
            twu[item] += transactionUtility;
            utility = static_cast<uint32_t>(std::stol(line.data() + i, &j));
            i += j + 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error parsing transaction: " << e.what() << std::endl;
        std::cerr << "Input line: " << line << std::endl;
        throw;
    }

    return transaction;
}

// Parse transactions from a file
auto parseTransactions(const std::string &inputPath) {
    auto startTime = std::chrono::steady_clock::now();

    std::unordered_map<std::string, uint32_t> item2id;
    std::unordered_map<uint32_t, std::string> id2item;
    std::vector<uint32_t> twu;

    int fd = open(inputPath.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error(strerror(errno));

#ifdef __linux__
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> transactions;
    constexpr std::size_t bufferSize = 4 * 1024;
    alignas(alignof(std::max_align_t)) char buffer[bufferSize];
    std::string line;

    while (auto bytesRead = read(fd, buffer, bufferSize)) {
        if (bytesRead < 0) throw std::runtime_error("Failed to read input file");

        char *prev = buffer, *current = nullptr;
        for (; (current = (char *)memchr(prev, '\n', buffer + bytesRead - prev)); prev = current + 1) {
            line.append(prev, current - prev);

            if (auto commentPos = line.find_first_of("%#@"); commentPos != std::string::npos)
                line.erase(commentPos);
            if (!line.empty()) {
                transactions.emplace_back(parseTransactionOneLine(std::move(line), item2id, id2item, twu));
                line.clear();
            }
        }
        line.append(prev, buffer + bytesRead - prev);
    }

    if (!line.empty()) {
        transactions.emplace_back(parseTransactionOneLine(std::move(line), item2id, id2item, twu));
    }

    close(fd);

    auto endTime = std::chrono::steady_clock::now();
    std::cout << "Parsing Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
              << " ms" << std::endl;

    return std::make_tuple(std::move(transactions), std::move(item2id), std::move(id2item), std::move(twu));
}
