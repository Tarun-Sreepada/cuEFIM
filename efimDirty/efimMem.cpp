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

#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <tuple>
#include <chrono>

#define kilo 1024ull
#define mega kilo * kilo
#define giga kilo * mega

// g++ -std=c++20 -o efimMem efimMem.cpp && ./efimMem test.txt space 40 output.txt


std::vector<std::pair<uint32_t, uint32_t>> parseTransactionOneLine(std::string line, auto &item2id, auto &id2item, auto &twu, const std::string &separator)
{
    std::vector<std::pair<uint32_t, uint32_t>> tra;
    uint32_t transaction_utility = 0;

    try
    {
        std::size_t i = 0, j = 0;
        std::vector<std::pair<uint32_t, uint32_t>> buf;
        while (i < line.size())
        {
            try
            {
                for (auto k = i; k < line.size(); ++k)
                {
                    if (line[k] == ' ' || line[k] == '\t' || line[k] == ':')
                    {
                        j = k - i;
                        break;
                    }
                }
                std::string result = line.substr(i, j);

                if (auto it = item2id.find(result); it != item2id.end())
                {
                    buf.push_back({it->second, 0});
                }
                else
                {
                    auto id = item2id.size();
                    item2id[result] = id;

                    id2item[id] = result;
                    buf.push_back({id, 0});
                    // twu.push_back(0);
                    twu.push_back({id, 0});
                }

                i += j + 1;
                if (line[i - 1] == ':')
                {
                    break;
                }
            }
            catch (std::invalid_argument &e)
            {
                std::cerr << e.what() << ":+ " << __LINE__ << " " << line << std::endl;
            }
        }

        tra.reserve(buf.size());

        for (auto &&e : buf)
            tra.push_back(std::move(e));

        try
        {
            transaction_utility = static_cast<uint32_t>(std::stol(line.data() + i, &j));
        }
        catch (std::invalid_argument &e)
        {
            std::cerr << e.what() << ":= " << __LINE__ << " " << line << std::endl;
        }

        i += j + 1;
        // std::cout << line[i - 1] << std::endl;
        if (line[i - 1] != ':')
        {
            // print line and i and line[i]
            std::cout << "|" << line.data() << std::endl;
            std::cout << "/";
            for (auto &t :tra)
            {
                std::cout << t.first << " ";
            }
            std::cout << std::endl;

            std::cout << "||" << line.data() + i << "||" << std::endl;
            throw std::runtime_error("failed to parse utils");
        }

        for (auto &[item, util] : tra)
        {
            // twu[item] += transaction_utility;
            twu[item].second += transaction_utility;
            try
            {
                util = static_cast<uint32_t>(std::stol(line.data() + i, &j));
                // std::cout << util << std::endl;
                i += j + 1;
            }
            catch (std::invalid_argument &e)
            {
                std::cerr << e.what() << ":- " << __LINE__ << " " << line << std::endl;
            }
        }
        // std::cout << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "input: " << line << std::endl;
        throw;
    }

    return std::move(tra);

    // return std::make_pair(std::move(tra), max_item);
}


auto parseTransactions(const std::string &input_path, const std::string &separator)
{
    std::unordered_map<std::string, uint32_t> item2id;
    std::unordered_map<uint32_t, std::string> id2item;
    std::vector<std::pair<uint32_t, uint32_t>> itemTWU;

    int fd = open(input_path.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error(strerror(errno));

    #if __linux__
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    #endif

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> transactions;
    std::string line;
    constexpr std::size_t buf_size = 4 * kilo;
    alignas(alignof(std::max_align_t)) char buf[buf_size];

    while (auto bytes_read = read(fd, buf, buf_size))
    {
        if (bytes_read < 0)
            throw std::runtime_error("read failed");

        char *prev = buf, *p = nullptr;
        for (; (p = (char *)memchr(prev, '\n', (buf + bytes_read) - prev)); prev = p + 1)
        {
            line.insert(line.size(), prev, p - prev);

            if (line.empty())
                continue;

            // std::cout << line << std::endl;
            auto transaction = parseTransactionOneLine(line, item2id, id2item, itemTWU, separator);
            line.clear();
            transactions.push_back(std::move(transaction));
        }
        line.insert(line.size(), prev, buf + buf_size - prev);
    }
    auto transaction = parseTransactionOneLine(line, item2id, id2item, itemTWU, separator);
    line.clear();
    transactions.push_back(std::move(transaction));

    close(fd);

    return std::make_tuple(std::move(transactions), std::move(item2id), std::move(id2item), std::move(itemTWU));

}

void printfile(auto &transactions, auto &item2id, auto &id2item, auto &itemTWU)
{
    for (auto &transaction : transactions)
    {
        for (auto &[item, util] : transaction)
        {
            std::cout << item << ":" << util << " ";
        }
        std::cout << std::endl;
    }

    for (auto &[item, twu] : itemTWU)
    {
        std::cout << item << " " << twu << std::endl;
    }
}

auto build(auto &transactions, auto &item2id, auto &id2item, auto &itemTWU, uint32_t minutil)
{
    // sort itemTWU
    std::sort(itemTWU.begin(), itemTWU.end(), [](auto &a, auto &b) {
        return a.second > b.second;
    });

    std::unordered_map<uint32_t, uint32_t> id2newid;
    std::unordered_map<uint32_t, std::string> newid2item;



    for (auto &[id, twu] : itemTWU)
    {
        if (twu < minutil)
        {
            break;
        }
        auto newid = newid2item.size() + 1;
        newid2item[newid] = id2item[id];
        id2newid[id] = newid;
    }

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> newtransactions;

    std::unordered_map<uint32_t, uint32_t> subtreeUtility(newid2item.size());
    std::unordered_map<uint32_t, uint32_t> localUtility(newid2item.size());

    uint32_t arrToAlloc = 0;

    for (auto &transaction : transactions)
    {
        std::vector<std::pair<uint32_t, uint32_t>> newtransaction;
        uint32_t sum = 0;
        for (auto &[item, util] : transaction)
        {
            if (auto it = id2newid.find(item); it != id2newid.end())
            {
                newtransaction.push_back({it->second, util});
                sum += util;
            }
        }
        // sort newtransaction by id
        std::sort(newtransaction.begin(), newtransaction.end(), [](auto &a, auto &b) {
            return a.first > b.first;
        });

        uint32_t temp = 0;

        for (auto &[item, util] : newtransaction)
        {
            subtreeUtility[item] += sum - temp;
            localUtility[item] += sum;
            temp += util;
        }

        arrToAlloc += newtransaction.size();

        newtransactions.push_back(std::move(newtransaction));
    }
    
    // sort newtransactions by size
    std::sort(newtransactions.begin(), newtransactions.end(), [](auto &a, auto &b) {
        return a.size() > b.size();
    });

    std::vector<uint32_t> primary;
    std::vector<uint32_t> secondary;

    for (uint32_t i = 0; i < subtreeUtility.size(); ++i)
    {
        if (subtreeUtility[i] >= minutil)
        {
            primary.push_back(i);
        }
        if (localUtility[i] >= minutil)
        {
            secondary.push_back(i);
        }
    }

    return std::make_tuple(std::move(newtransactions), std::move(newid2item), std::move(primary), std::move(secondary), arrToAlloc);

}

uint32_t *getArray(uint32_t *workspace, uint32_t size)
{

    uint32_t index = workspace[0];
    if (index + size > 2 * giga)
    {
        std::cerr << "Out of memory workspace" << std::endl;
        exit(EXIT_FAILURE);
    }
    workspace[0] += size;

    return &workspace[index];
}

std::tuple<uint *, uint *, uint *, uint *, 
                uint , uint *, uint *, uint *, uint *, uint32_t *> 
toArrays(auto &transactions, auto &primary, auto &secondary, uint32_t arrToAlloc)
{
    uint32_t *workspace = new uint32_t[giga * 2];
    memset(workspace, 0, 2 * giga * sizeof(uint32_t));
    workspace[0] = 1;

    // create pattern
    uint32_t *pattern = getArray(workspace, 1);
    pattern[0] = 0;

    // std::cout << "Pattern Length: " << pattern[0] << std::endl;


    // create primary
    uint32_t *arrPrimary = getArray(workspace, primary.size() + 1);
    arrPrimary[0] = primary.size();
    for (uint32_t i = 0; i < primary.size(); ++i)
    {
        arrPrimary[i + 1] = primary[i];
    }
    // create secondary
    uint32_t *arrSecondary = getArray(workspace, secondary.size() + 1);
    
    arrSecondary[0] = secondary.size();
    for (uint32_t i = 0; i < secondary.size(); ++i)
    {
        arrSecondary[i + 1] = secondary[i];
    }


    // create transactions
    uint32_t *numTransactions = getArray(workspace, 1);
    numTransactions[0] = transactions.size();
    uint32_t *indexStart = getArray(workspace, transactions.size());
    uint32_t *indexEnd = getArray(workspace, transactions.size());
    uint32_t *costs = getArray(workspace, transactions.size());
    uint32_t *itemStart = getArray(workspace, arrToAlloc);
    uint32_t *utilStart = getArray(workspace, arrToAlloc);


    uint32_t index = 0;
    for (uint32_t i = 0; i < transactions.size(); ++i)
    {
        indexStart[i] = index;
        for (auto &[item, util] : transactions[i])
        {
            itemStart[index] = item;
            utilStart[index] = util;
            index++;
        }
        indexEnd[i] = index;
    }

    return std::make_tuple(workspace, pattern, arrPrimary, arrSecondary, numTransactions[0], indexStart, indexEnd, costs, itemStart, utilStart);

}

void printProcessed(auto &transactions, auto &id2item, auto &primary, auto &secondary, uint32_t arrToAlloc)
{
    std::cout << "Primary: ";
    for (auto &p : primary)
    {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    std::cout << "Secondary: ";
    for (auto &s : secondary)
    {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::cout << "Transactions: " << transactions.size() << std::endl;
    for (auto &transaction : transactions)
    {
        for (auto &[item, util] : transaction)
        {
            std::cout << item << ":" << util << " ";
        }
        std::cout << std::endl;
    }

}


void addPattern(uint32_t *pattern, uint32_t minutil, uint32_t *patterns)
{
    uint32_t loc = patterns[0];
    if (loc + pattern[0] + 2 > giga)
    {
        std::cerr << "Out of memory patterns" << std::endl;
        exit(EXIT_FAILURE);
    }
    patterns[0] += pattern[0] + 2; // pat + minutil + 0
    patterns[1]++; // increment number of patterns

    for (uint32_t i = 0; i < pattern[0]; ++i)
    {
        patterns[loc + i] = pattern[i + 1];
    }
    patterns[loc + pattern[0]] = minutil;

}



bool isInSecondary(auto &secondary, auto item)
{
    uint32_t size = secondary[0];
    for (uint32_t i = 0; i < size; ++i)
    {
        if (secondary[1 + i] == item)
        {
            return true;
        }
    }

    return false;
}

void createLocalAndSubtreeUtil(auto &workspace, auto &collection, auto &localUtil, auto &subtreeUtil)
{
    uint32_t numSecondary = workspace[collection[2]];

    for (uint32_t i = 0; i < numSecondary; ++i)
    {
        localUtil[i * 2] = workspace[collection[2] + 1 + i];
        localUtil[i * 2 + 1] = 0;
        subtreeUtil[i * 2] = workspace[collection[2] + 1 + i];
        subtreeUtil[i * 2 + 1] = 0;
    }
}

// addLocalAndSubtreeUtil(secondary, localUtil, subtreeUtil, itemStart[k], &transactionSumUtil[j], utilStart[k]);
void addLocalAndSubtreeUtil(auto &secondary, auto &localUtil, auto &subtreeUtil, auto item, auto transactionSumUtil, auto tempUtil)
{
    uint32_t size = secondary[0];
    for (uint32_t i = 0; i < size; i++)
    {
        if (secondary[i + 1] == item)
        {
            localUtil[i] += transactionSumUtil;
            subtreeUtil[i] += transactionSumUtil - tempUtil;
            break;
        }
    }
}

uint32_t *writeNewPattern(uint32_t *workspace, uint32_t *pattern, uint32_t item)
{
    if (pattern[0] == 0)
    {
        uint32_t *newPattern = getArray(workspace, 2);
        newPattern[0] = 1;
        newPattern[1] = item;
        return &newPattern[0];
    }
    uint32_t *newPattern = getArray(workspace, pattern[0] + 2);
    newPattern[0] = pattern[0] + 1;
    for (uint32_t i = 0; i < pattern[0]; ++i)
    {
        newPattern[i + 1] = pattern[i + 1];
    }
    newPattern[pattern[0] + 1] = item;

    return &newPattern[0];

}

void printArray(auto &workspace, auto &pattern, auto &primary, auto &secondary, auto numTransactions, auto &indexStart, auto &indexEnd, auto &costs, auto &itemStart, auto &utilStart)
{

    std::cout << "pattern: ";
    for (uint32_t i = 0; i < pattern[0]; ++i)
    {
        std::cout << pattern[i + 1] << " ";
    }
    std::cout << std::endl;
    std::cout << "Pattern Length: " << pattern[0] << std::endl;

    std::cout << "primary: ";
    for (uint32_t i = 0; i < primary[0]; ++i)
    {
        std::cout << primary[i + 1] << " ";
    }
    std::cout << std::endl;

    std::cout << "secondary: ";
    for (uint32_t i = 0; i < secondary[0]; ++i)
    {
        std::cout << secondary[i + 1] << " ";
    }
    std::cout << std::endl;

    std::cout << "numTransactions: " << numTransactions << std::endl;

    for (uint32_t i = 0; i < numTransactions; ++i)
    {
        std::cout << "Cost: " << costs[i] << "\t|" << indexStart[i] << "|";
        for (uint32_t j = indexStart[i]; j < indexEnd[i]; ++j)
        {
            std::cout << itemStart[j] << ":" << utilStart[j] << " ";
        }
        std::cout << "|" << indexEnd[i] << std::endl;
    }
}


void printanArray(auto &arr, auto size)
{
    for (uint32_t i = 0; i < size; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void mine(auto &workspace, auto &pattern, auto &primary, auto &secondary, 
            auto numTransactions, auto &indexStart, auto &indexEnd, auto &costs, 
            auto &itemStart, auto &utilStart, auto &patterns, auto minUtil, int depth)
{
    for (uint32_t i = 1; i <= primary[0]; i++) // iterate over cands
    {

        uint32_t *newPattern = writeNewPattern(workspace, pattern, primary[i]);
        uint32_t newPatternCost = 0;

        uint32_t *projectedIndexStart = getArray(workspace, numTransactions); // no need for end because it is the same as indexEnd
        uint32_t *projectedCosts = getArray(workspace, numTransactions);

        uint32_t numNewItemsToAlloc = 0;
        uint32_t numNewTrans = 0;

        uint32_t *transactionSumUtil = getArray(workspace, numTransactions);

        // iterate over transactions
        for (uint32_t j = 0; j < numTransactions; j++)
        {
            uint32_t found = 0;
            for (uint32_t k = indexStart[j]; k < indexEnd[j]; k++)
            {
                if (found == 1)
                {
                    if (isInSecondary(secondary, itemStart[k]))
                    {
                        transactionSumUtil[j] += utilStart[k];
                        numNewItemsToAlloc++;
                    }
                }
                if (itemStart[k] == primary[i])
                {
                    projectedIndexStart[j] = k + 1;
                    projectedCosts[j] += utilStart[k] + costs[j];
                    newPatternCost += utilStart[k] + costs[j];
                    found = 1;
                    numNewTrans++;
                    transactionSumUtil[j] = projectedCosts[j];
                }
                
            }
            if (found == 0)
            {
                projectedIndexStart[j] = indexEnd[j];
                projectedCosts[j] = 0;
            }
        }


        if (newPatternCost >= minUtil)
        {
            addPattern(newPattern, newPatternCost, patterns);
        }

        if (numNewItemsToAlloc == 0)
        {
            continue;
        }

        uint32_t *newItems = getArray(workspace, numNewItemsToAlloc);
        uint32_t *newUtils = getArray(workspace, numNewItemsToAlloc);
        uint32_t *newIndexStart = getArray(workspace, numNewTrans);
        uint32_t *newIndexEnd = getArray(workspace, numNewTrans);
        uint32_t *newCosts = getArray(workspace, numNewTrans);

        uint32_t *localUtil = getArray(workspace, secondary[0]);
        uint32_t *subtreeUtil = getArray(workspace, secondary[0]);

        uint32_t index = 0;
        uint32_t transIndex = 0;



        for (uint32_t j = 0; j < numTransactions; j++)
        {
            if (projectedIndexStart[j] < indexEnd[j])
            {
                uint32_t tempUtil = 0;
                bool flag = false;
                newCosts[transIndex] = projectedCosts[j];

                for (uint32_t k = projectedIndexStart[j]; k < indexEnd[j]; k++)
                {
                    if (isInSecondary(secondary, itemStart[k]))
                    {
                        newItems[index] = itemStart[k];
                        newUtils[index] = utilStart[k];
                        index++;
                        addLocalAndSubtreeUtil(secondary, localUtil, subtreeUtil, itemStart[k], transactionSumUtil[j], tempUtil);
                        tempUtil += utilStart[k];
                        if (!flag)
                        {
                            flag = true;
                            newIndexStart[transIndex] = index - 1;
                        }

                    }
                }
                newIndexEnd[transIndex] = index;
                transIndex++;
            
            }
            
        }


        uint32_t numNewSecondary = 0;
        uint32_t numNewPrimary = 0;

        for (uint32_t j = 0; j < secondary[0]; j++)
        {
            if (localUtil[j] >= minUtil)
            {
                numNewSecondary++;
            }
            if (subtreeUtil[j] >= minUtil)
            {
                numNewPrimary++;
            }
        }

        uint32_t *newPrimary = getArray(workspace, numNewPrimary + 1);
        uint32_t *newSecondary = getArray(workspace, numNewSecondary + 1);
        newPrimary[0] = numNewPrimary;
        newSecondary[0] = numNewSecondary;

        numNewPrimary = 1;
        numNewSecondary = 1;

        for (uint32_t j = 0; j < secondary[0]; j++)
        {
            if (subtreeUtil[j] >= minUtil)
            {
                newPrimary[numNewPrimary++] = secondary[j + 1];
            }
            if (localUtil[j] >= minUtil)
            {
                newSecondary[numNewSecondary++] = secondary[j + 1];
            }
        }


        if (newPrimary[0])
        {
            mine(workspace, newPattern, newPrimary, newSecondary, numNewTrans, newIndexStart, newIndexEnd, newCosts, newItems, newUtils, patterns, minUtil, depth + 1);
        }


    }
}


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Wrong number of arguments. Expected 4, got " << argc - 1 << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input file> <minutil> <output file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string inputFileName;
    uint32_t minutil;
    std::string separator = ",";
    std::string outputFileName;

    try
    {
        inputFileName = argv[1];
        minutil = std::stoi(argv[3]);
        outputFileName = argv[4];

        // separator handling
        if (std::string(argv[2]) == "tab")
        {
            separator = "\t";
        }
        else if (std::string(argv[2]) == "comma")
        {
            separator = ",";
        }
        else if (std::string(argv[2]) == "space")
        {
            separator = " ";
        }
        else{
            std::cerr << "Invalid separator. Expected 'tab' or 'comma' or 'space', got " << argv[2] << std::endl;
        }
    }
    catch (std::invalid_argument &e)
    {
        // print error message and usage
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input file> <separator> <minutil> <output file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Input file: " << inputFileName << std::endl;
    std::cout << "Minutil: " << minutil << std::endl;
    std::cout << "Separator: " << separator << std::endl;
    std::cout << "Output file: " << outputFileName << std::endl;

    // start clock
    auto start = std::chrono::high_resolution_clock::now();

    // read file
    auto [transactions, item2id, id2item, itemTWU] = parseTransactions(inputFileName, separator);

    // print file
    // printfile(transactions, item2id, id2item, itemTWU);

    // process file
    auto [newtransactions, newid2item, primaryV, secondaryV, arrToAlloc] = build(transactions, item2id, id2item, itemTWU, minutil);

    //print file
    // printProcessed(newtransactions, newid2item, primaryV, secondaryV, arrToAlloc);

    auto [workspace, pattern, primary, secondary, 
    numTransactions, indexStart, indexEnd, costs, itemStart, utilStart] = toArrays(newtransactions, primaryV, secondaryV, arrToAlloc);


    uint32_t patterns[mega];
    patterns[0] = 2;    

    std::cout << std::endl;
    mine(workspace, pattern, primary, secondary, numTransactions, indexStart, indexEnd, costs, itemStart, utilStart, patterns, minutil, 0);

    std::cout << "Num Patterns: " << patterns[1] << std::endl;
    std::vector<uint32_t> pat;
    for (uint32_t i = 0; i < patterns[0]; i++)
    {
        if (patterns[i + 2] != 0)
        {
            pat.push_back(patterns[i + 2]);
        }
        if (patterns[i + 2] == 0)
        {
            if (!pat.size()) continue;
            for (uint32_t j = 0; j < pat.size() - 1; j++)
            {
                std::cout << newid2item[pat[j]] << " ";
            }
            std::cout << "#UTIL: " << pat[pat.size() - 1] << std::endl;

            pat.clear();
        }

    }
    // return SUCCESS
    return 0;

}
