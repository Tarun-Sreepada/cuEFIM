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

// Function to parse line components
bool parse_line_components(const std::string &line, std::string &keys_str, std::string &weight_str, std::string &values_str) {
    size_t pos1 = line.find(':');
    if (pos1 == std::string::npos) return false;

    size_t pos2 = line.find(':', pos1 + 1);
    if (pos2 == std::string::npos) return false;

    keys_str = line.substr(0, pos1);
    weight_str = line.substr(pos1 + 1, pos2 - pos1 - 1);
    values_str = line.substr(pos2 + 1);

    return !keys_str.empty() && !weight_str.empty() && !values_str.empty();
}


// Function to parse keys and update 'twu'
void parse_keys(const std::string &keys_str, char separator_char, uint32_t weight,
                std::unordered_map<std::string, uint32_t> &twu, std::vector<std::string> &keys) {
    size_t start = 0;
    size_t end = keys_str.find(separator_char);
    while (end != std::string::npos) {
        std::string key = keys_str.substr(start, end - start);
        keys.push_back(key);
        twu[key] += weight;
        start = end + 1;
        end = keys_str.find(separator_char, start);
    }
    // Add the last key
    std::string key = keys_str.substr(start);
    keys.push_back(key);
    twu[key] += weight;
}



bool parse_values(const std::string &values_str, char separator_char, std::vector<uint32_t> &values) {
    size_t start = 0;
    size_t end = values_str.find(separator_char);
    while (end != std::string::npos) {
        std::string_view value_str = std::string_view(values_str).substr(start, end - start);
        try {
            values.push_back(std::stoul(std::string(value_str)));
        } catch (const std::exception &e) {
            std::cerr << "Invalid value: " << value_str << " -> " << e.what() << std::endl;
            return false;
        }
        start = end + 1;
        end = values_str.find(separator_char, start);
    }
    // Add the last value
    std::string_view value_str = std::string_view(values_str).substr(start);
    try {
        values.push_back(std::stoul(std::string(value_str)));
    } catch (const std::exception &e) {
        std::cerr << "Invalid value: " << value_str << " -> " << e.what() << std::endl;
        return false;
    }
    return true;
}


// Function to process a single line
bool process_line(const std::string &line, char separator_char,
                  std::unordered_map<std::string, uint32_t> &twu,
                  line_data &ld) {
    std::string keys_str, weight_str, values_str;
    if (!parse_line_components(line, keys_str, weight_str, values_str)) {
        std::cerr << "Skipping malformed line: " << line << std::endl;
        return false;
    }

    uint32_t weight;
    try {
        weight = std::stoull(weight_str);
    } catch (const std::exception &e) {
        std::cerr << "Invalid weight in line: " << line << " -> " << e.what() << std::endl;
        return false;
    }

    std::vector<std::string> keys;
    parse_keys(keys_str, separator_char, weight, twu, keys);

    std::vector<uint32_t> values;
    if (!parse_values(values_str, separator_char, values)) {
        return false;
    }

    if (keys.size() != values.size()) {
        std::cerr << "Error: Mismatch between keys and values in line: " << line << std::endl;
        return false;
    }

    ld.transaction.clear();
    for (size_t i = 0; i < keys.size(); ++i) {
        ld.transaction.push_back({ keys[i], values[i] });
    }

    return true;
}

file_data read_file(const params &p, std::unordered_map<std::string, uint32_t> &twu, char separator_char) {
    file_data file;
    std::ifstream input(p.input_file);

    if (!input.is_open()) {
        std::cerr << "Error opening file: " << strerror(errno) << std::endl;
        exit(1);
    }

    std::ios::sync_with_stdio(false); // Untie C++ streams from C streams
    std::string line;

    while (std::getline(input, line)) {
        line_data ld;
        if (process_line(line, separator_char, twu, ld)) {
            file.data.emplace_back(std::move(ld));
        }
    }
    input.close();
    return file;
}


std::vector<item_utility> filter_and_sort_twu(const std::unordered_map<std::string, uint32_t> &twu, uint32_t min_utility) {
    std::vector<item_utility> sorted_twu;
    for (const auto &entry : twu) {
        if (entry.second >= min_utility) {
            sorted_twu.push_back({entry.first, entry.second});
        }
    }

    std::sort(sorted_twu.begin(), sorted_twu.end(), [](const auto &a, const auto &b) {
        return a.utility > b.utility;
    });

    return sorted_twu;
}


// Helper function to build a transaction from line data
std::vector<key_value> build_transaction(
    const line_data& data,
    const std::unordered_map<std::string, uint32_t>& strToInt)
{
    std::vector<key_value> transaction;
    for (const auto& entry : data.transaction) {
        auto it = strToInt.find(entry.item);
        if (it != strToInt.end()) {
            transaction.push_back({ it->second, entry.utility });
        }
    }
    return transaction;
}

// Helper function to update utility maps
void update_utilities(
    const std::vector<key_value>& transaction,
    std::unordered_map<uint32_t, uint32_t>& subtree_util)
{
    uint32_t weight = std::accumulate(transaction.begin(), transaction.end(), uint32_t(0),
                                      [](uint32_t sum, const auto& entry) {
                                          return sum + entry.value;
                                      });
    uint32_t temp = 0;
    for (const auto& entry : transaction) {
        uint32_t item_id = entry.key;
        uint32_t value = entry.value;
        subtree_util[item_id] += weight - temp;
        temp += value;
    }
}


// Helper function to collect primary and secondary items
void collect_primary_secondary(
    const std::unordered_map<uint32_t, uint32_t>& subtree_util,
    uint32_t min_utility,
    std::vector<uint32_t>& primary)
{
    for (const auto& entry : subtree_util) {
        if (entry.second >= min_utility) {
            primary.push_back(entry.first);
        }
    }
}

// Main function to process transactions
std::tuple<
    std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash>,
    std::vector<uint32_t>,
    std::vector<uint32_t>
> process_transactions(
    const file_data &fd,
    const std::unordered_map<std::string, uint32_t>& strToInt,
    uint32_t min_utility)
{
    std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash> filtered_transactions;
    std::unordered_map<uint32_t, uint32_t> subtree_util;

    for (const auto& data : fd.data) {
        

        // Build transaction
        auto transaction = build_transaction(data, strToInt);

        if (!transaction.empty()) {
            // Sort transaction by item IDs
            std::sort(transaction.begin(), transaction.end(), [](const auto& a, const auto& b) {
                return a.key < b.key;
            });


            // Separate keys and values
            std::vector<uint32_t> transaction_keys;
            std::vector<uint32_t> transaction_values;
            for (const auto& entry : transaction) {
                transaction_keys.push_back(entry.key);
                transaction_values.push_back(entry.value);
            }

            // Update utilities
            update_utilities(transaction, subtree_util);

            // update_filtered_transactions(transaction_keys, transaction_values, filtered_transactions);
            auto it = filtered_transactions.find(transaction_keys);
            if (it != filtered_transactions.end()) {
                for (size_t i = 0; i < transaction_values.size(); ++i) {
                    it->second[i] += transaction_values[i];
                }
            } else {
                filtered_transactions[transaction_keys] = transaction_values;
            }

        }
    }

    // Collect primary and secondary items
    std::vector<uint32_t> primary;
    std::vector<uint32_t> secondary;
    collect_primary_secondary(subtree_util, min_utility, primary);

    return std::make_tuple(filtered_transactions, primary, secondary);
}

// Helper function to map items to integer IDs
void map_items_to_ids(
    const std::vector<item_utility>& sorted_twu,
    std::unordered_map<std::string, uint32_t>& strToInt,
    std::unordered_map<uint32_t, std::string>& intToStr)
{
    uint32_t id = 1;
    strToInt.reserve(sorted_twu.size());
    intToStr.reserve(sorted_twu.size());

    for (const auto& entry : sorted_twu) {
        const std::string& item = entry.item;
        intToStr[id] = item;
        strToInt[item] = id++;
    }
}


// Main function to parse the file
std::tuple<
    std::unordered_map<std::vector<uint32_t>, std::vector<uint32_t>, VectorHash>,
    std::vector<uint32_t>,
    std::vector<uint32_t>,
    std::unordered_map<uint32_t, std::string>> 
parse_file(const params& p)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    char separator_char = get_separator(p.separator);

    // Read file and calculate TWU
    std::unordered_map<std::string, uint32_t> twu;
    auto file_data = read_file(p, twu, separator_char);
    std::cout << "File read in: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    // Filter and sort TWU
    auto sorted_twu = filter_and_sort_twu(twu, p.min_utility);

    // Map items to integer IDs
    std::unordered_map<std::string, uint32_t> strToInt;
    std::unordered_map<uint32_t, std::string> intToStr;
    map_items_to_ids(sorted_twu, strToInt, intToStr);

    std::cout << "Items mapped in: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    // Process transactions
    auto [filtered_transactions, primary, secondary] = process_transactions(file_data, strToInt, p.min_utility);
    secondary.assign(intToStr.size() + 1, 1);

    std::cout << "Transactions processed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    return { std::move(filtered_transactions), std::move(primary), std::move(secondary), std::move(intToStr) };
}
