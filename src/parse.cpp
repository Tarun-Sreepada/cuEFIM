#include "parse.hpp"

parsed_file parse_file_cpu(raw_file &file, results &r, Config::Params &p)
{
    uint32_t total_utility = 0;

    parsed_file pf;
    std::vector<uint32_t> items;
    std::vector<uint32_t> utilities;
    uint32_t transaction_utility = 0;

    enum ParseState
    {
        PARSE_ITEMS,
        PARSE_TOTAL_UTILITY,
        PARSE_UTILITIES
    };
    ParseState state = PARSE_ITEMS;

    while (file.processed_bytes < file.retrieved_bytes)
    {

        size_t end = file.processed_bytes;
        // Find next separator or delimiter
        while (end < file.retrieved_bytes && file.data[end] != p.separator_char && file.data[end] != ':' && file.data[end] != '\n')
        {
            end++;
        }

        if (end == file.processed_bytes) // Empty segment, move to the next character
        {
            file.processed_bytes++;
            continue;
        }

        uint32_t value = 0;
        auto result = std::from_chars(file.data + file.processed_bytes, file.data + end, value);
        if (result.ec != std::errc())
        {
            std::cerr << "Error parsing value at position " << file.processed_bytes << "\n";
            exit(1);
        }

        switch (state)
        {
        case PARSE_ITEMS:
            items.push_back(value);
            if (file.data[end] == ':') // End of items section
            {
                state = PARSE_TOTAL_UTILITY;
            }
            break;

        case PARSE_TOTAL_UTILITY:
            total_utility = value;
            state = PARSE_UTILITIES;
            break;

        case PARSE_UTILITIES:
            utilities.push_back(value);
            break;
        }

        // Move to the next segment
        file.processed_bytes = end + 1;

        // Break if newline or end of buffer is reached
        if (end >= file.retrieved_bytes || file.data[end] == '\n') // Safe bounds check
        {
            // Check if we have processed items correctly
            if (items.size() != utilities.size())
            {
                std::cerr << "\nError: Item count does not match item utility count\n";
                std::cerr << "Item count: " << items.size() << " Utility count: " << utilities.size() << std::endl;
                exit(1);
            }

            // Reset for the next transaction
            state = PARSE_ITEMS;

            // update twu
            for (size_t i = 0; i < items.size(); i++)
            {
                pf.twu[items[i]] += total_utility;
            }

            pf.key_value_pairs.push_back(std::make_pair(std::move(items), std::move(utilities)));

            items.clear();
            utilities.clear();
        }
    }

    r.record_memory_usage("File Parse");

    return pf;
}
