#include "build.hpp"

build_file build_cpu(parsed_file &pf, results &r, Config::Params &p)
{
    build_file bf;

    // make ordered twu in descending order unless less tha minutil then break
    for (const auto &item : pf.twu)
    {
        if (item.second < p.min_utility)
        {
            continue;
        }
        bf.ordered_twu[item.first] = item.second;
    }

    bf.ordered_twu_vector = std::vector<std::pair<uint32_t, uint32_t>>(bf.ordered_twu.begin(), bf.ordered_twu.end());
    std::sort(bf.ordered_twu_vector.begin(), bf.ordered_twu_vector.end(),
              [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
                  return a.second > b.second;
              });

    // create item to itemID mapping start from 1
    uint32_t itemID = bf.ordered_twu.size();
    for (const auto &item : bf.ordered_twu_vector)
    {
        bf.itemID_to_item[itemID] = item.first;
        bf.item_to_itemID[item.first] = itemID--;
    }

    for (const auto &transaction : pf.key_value_pairs)
    {
        std::vector<key_value> temp;
        uint32_t total_value = 0;
        for (size_t i = 0; i < transaction.first.size(); i++)
        {
            if (bf.item_to_itemID.find(transaction.first[i]) != bf.item_to_itemID.end())
            {
                temp.push_back({bf.item_to_itemID[transaction.first[i]], transaction.second[i]});
                total_value += transaction.second[i];
            }
        }

        if (temp.empty())
        {
            continue;
        }

        bf.csr_transaction_start.push_back(bf.total_items);

        std::sort(temp.begin(), temp.end(),
                  [](const key_value &a, const key_value &b) {
                      return a.key < b.key;
                  });

        uint32_t temp_val = 0;
        for (size_t i = 0; i < temp.size(); i++)
        {
            bf.subtree_utility[temp[i].key] += total_value - temp_val;
            temp_val += temp[i].value;
        }

        bf.max_transaction_size = std::max(bf.max_transaction_size, temp.size());
        bf.compressed_spare_row_db.insert(bf.compressed_spare_row_db.end(), temp.begin(), temp.end());
        bf.total_items += temp.size();
        bf.csr_transaction_end.push_back(bf.total_items);
        bf.transaction_count++;
    }

    // create primary and secondary
    for (const auto &item : bf.subtree_utility)
    {
        if (item.second >= p.min_utility)
        {   
            bf.primary.push_back(item.first);
        }
    }

    bf.secondary.assign(bf.itemID_to_item.size() + 1, 1);

    r.record_memory_usage("Build");

    return std::move(bf);
}
