#include "config.cuh"
#include "parse.cuh"

// Function to determine the separator character
char get_separator(const std::string &separator)
{
    if (separator == "space")
        return ' ';
    if (separator == "tab")
        return '\t';
    return ',';
}

std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> read_file(const params &p, std::unordered_map<std::string, uint64_t> &twu, char separator_char)
{
    std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> file_data;
    std::ifstream input(p.input_file);

    if (!input.is_open())
    {
        std::cerr << "Error opening file: " << strerror(errno) << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(input, line))
    {
        try
        {
            std::istringstream iss(line);
            std::string keys_str, weight_str, values_str;

            std::getline(iss, keys_str, ':');
            std::getline(iss, weight_str, ':');
            std::getline(iss, values_str, ':');

            if (keys_str.empty() || weight_str.empty() || values_str.empty())
            {
                std::cerr << "Skipping malformed line: " << line << std::endl;
                continue;
            }

            uint64_t weight = std::stoull(weight_str);
            std::vector<std::string> keys;
            std::vector<uint64_t> values;

            std::istringstream keys_stream(keys_str);
            std::string key;
            while (std::getline(keys_stream, key, separator_char))
            {
                keys.push_back(key);
                twu[key] += weight;
            }

            std::istringstream values_stream(values_str);
            std::string value;
            while (std::getline(values_stream, value, separator_char))
            {
                values.push_back(std::stoull(value));
            }

            if (keys.size() != values.size())
            {
                std::cerr << "Error: Mismatch between keys and values in line: " << line << std::endl;
                continue;
            }

            file_data.emplace_back(keys, values);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing line: " << line << " -> " << e.what() << std::endl;
        }
    }
    input.close();
    return file_data;
}

std::vector<std::pair<std::string, uint64_t>> filter_and_sort_twu(const std::unordered_map<std::string, uint64_t> &twu, uint64_t min_utility)
{
    std::vector<std::pair<std::string, uint64_t>> sorted_twu;
    for (const auto &entry : twu)
    {
        if (entry.second >= min_utility)
        {
            sorted_twu.push_back(entry);
        }
    }

    std::sort(sorted_twu.begin(), sorted_twu.end(), [](const auto &a, const auto &b)
              { return a.second < b.second; });

    return sorted_twu;
}

std::tuple<
    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>,
    std::unordered_map<uint64_t, uint64_t>,
    std::unordered_map<uint64_t, uint64_t>
> process_transactions
(
    const std::vector<std::pair<std::vector<std::string>, std::vector<uint64_t>>> &file_data,
    const std::unordered_map<std::string, uint32_t> &strToInt, uint64_t min_utility)
{
    // items, values, weight    0 for initial wegiht we use this later
    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash> filtered_transactions;

    std::unordered_map<uint64_t, uint64_t> subtree_util;
    std::unordered_map<uint64_t, uint64_t> secondary_util;

    for (const auto &data : file_data)
    {
        const auto &keys = data.first;
        const auto &values = data.second;

        std::vector<std::pair<uint32_t, uint64_t>> transaction;

        for (size_t i = 0; i < keys.size(); ++i)
        {
            if (strToInt.count(keys[i]))
            {
                transaction.emplace_back(strToInt.at(keys[i]), values[i]);
            }
        }

        if (!transaction.empty())
        {
            std::sort(transaction.begin(), transaction.end(), [](const auto &a, const auto &b)
                      { return a.first < b.first; });

            std::vector<uint32_t> transaction_keys;
            std::vector<uint64_t> transaction_values;

            uint64_t weight = std::accumulate(transaction.begin(), transaction.end(), 0, [](uint64_t sum, const auto &entry)
                                              { return sum + entry.second; });
            uint64_t temp = 0;

            for (const auto &entry : transaction)
            {
                transaction_keys.push_back(entry.first);
                transaction_values.push_back(entry.second);
                subtree_util[entry.first] += weight - temp;
                secondary_util[entry.first] += weight;
                temp += entry.second;
            }

            auto &entry = filtered_transactions[transaction_keys];
            if (entry.first.empty())
            {
                entry.first = transaction_values;
            }
            else
            {
                for (size_t i = 0; i < transaction_values.size(); ++i)
                {
                    entry.first[i] += transaction_values[i];
                }
            }
        }
    }

    return std::make_tuple(filtered_transactions, subtree_util, secondary_util);
}

std::tuple<
    std::unordered_map<std::string, uint32_t>,
    std::unordered_map<uint32_t, std::string>,
    std::unordered_map<std::vector<uint32_t>, std::pair<std::vector<uint64_t>, uint64_t>, VectorHash>,
    std::vector<std::pair<std::string, uint64_t>>,
    std::unordered_map<uint64_t, uint64_t>,
    std::unordered_map<uint64_t, uint64_t>
>
 parse_file(const params &p)
{
    char separator_char = get_separator(p.separator);

    // Read file and calculate TWU
    std::unordered_map<std::string, uint64_t> twu;
    auto file_data = read_file(p, twu, separator_char);

    // Filter and sort TWU
    auto sorted_twu = filter_and_sort_twu(twu, p.min_utility);

    // Map items to integer IDs
    std::unordered_map<std::string, uint32_t> strToInt;
    std::unordered_map<uint32_t, std::string> intToStr;
    uint32_t id = 1;
    for (const auto &entry : sorted_twu)
    {
        intToStr[id] = entry.first;
        strToInt[entry.first] = id++;
    }

    // Process transactions
    auto [filtered_transactions, subtree_util, secondary_util] = process_transactions(file_data, strToInt, p.min_utility);

    // return strToInt, intToStr, filtered_transactions, sorted_twu, primary and secondary
    return std::make_tuple(strToInt, intToStr, filtered_transactions, sorted_twu, subtree_util, secondary_util);
}