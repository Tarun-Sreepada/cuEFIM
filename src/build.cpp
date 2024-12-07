#include "build.hpp"


// // Define a utility function to sort the TWU vector and create item ranking
// std::unordered_map<uint32_t, uint32_t> create_item_ranking(const std::unordered_map<uint32_t, uint32_t> &twu)
// {
//     std::vector<std::pair<uint32_t, uint32_t>> twu_vector(twu.begin(), twu.end());

//     // Sort items by descending TWU value
//     std::sort(twu_vector.begin(), twu_vector.end(),
//               [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
//                   return a.second < b.second; // Descending order
//               });

//     // Create a mapping from item to its rank
//     std::unordered_map<uint32_t, uint32_t> item_rank;
//     for (size_t i = 0; i < twu_vector.size(); i++)
//     {
//         item_rank[twu_vector[i].first] = i + 1;
//     }

//     return item_rank;
// }

// // Define a utility function to filter transactions based on minimum utility
// size_t filter_transaction(database &db, size_t index, const std::unordered_map<uint32_t, uint32_t> &twu, double minutil)
// {
//     size_t start = db.csr_transaction_start[index];
//     size_t end = db.csr_transaction_end[index];

//     size_t new_start = start;
//     while (new_start < end && twu.at(db.compressed_spare_row_db[new_start].key) < minutil)
//     {
//         new_start++;
//     }

//     db.csr_transaction_start[index] = new_start;
//     return new_start;
// }

// // Define a utility function to compute subtree utilities
// void compute_subtree_utilities(database &db, size_t start, size_t end, std::unordered_map<uint32_t, uint32_t> &subtree_utility)
// {
//     uint32_t utility = 0;
//     for (size_t j = start; j < end; j++)
//     {
//         utility += db.compressed_spare_row_db[j].value;
//     }

//     uint32_t temp = 0;
//     for (size_t j = start; j < end; j++)
//     {
//         subtree_utility[db.compressed_spare_row_db[j].key] += utility - temp;
//         temp += db.compressed_spare_row_db[j].value;
//     }
// }

// // Define a utility function to remove empty transactions
// void remove_empty_transactions(database &db, std::vector<size_t> &indices_to_remove)
// {
//     for (size_t index : indices_to_remove)
//     {
//         db.csr_transaction_start.erase(db.csr_transaction_start.begin() + index);
//         db.csr_transaction_end.erase(db.csr_transaction_end.begin() + index);
//         db.db_size--;
//     }
// }

// // Main build function
// std::tuple<std::unordered_map<uint32_t, uint32_t>, std::unordered_map<uint32_t, uint32_t>>
//     build(database &db, std::unordered_map<uint32_t, uint32_t> &twu, double minutil, results &r)
// {
//     #ifdef DEBUG
//         std::cout << "Database: " << db.db_size << " transactions" << std::endl;
//         for (size_t i = 0; i < db.db_size; i++)
//         {
//             for (size_t j = db.csr_transaction_start[i]; j < db.csr_transaction_end[i]; j++)
//             {
//                 std::cout << db.compressed_spare_row_db[j].key << ":" << db.compressed_spare_row_db[j].value << " ";
//             }
//             std::cout << std::endl;
//         }
//     #endif

//     // Create item ranking
//     auto item_rank = create_item_ranking(twu);

//     // Initialize subtree utility and indices for removal
//     std::unordered_map<uint32_t, uint32_t> subtree_utility;
//     std::vector<size_t> indices_to_remove;

//     // Process transactions
//     for (size_t i = 0; i < db.db_size; i++)
//     {
//         size_t start = db.csr_transaction_start[i];
//         size_t end = db.csr_transaction_end[i];

//         // Sort items in transaction by rank
//         std::sort(db.compressed_spare_row_db.begin() + start, db.compressed_spare_row_db.begin() + end,
//                   [&](const key_value &a, const key_value &b) {
//                       return item_rank[a.key] < item_rank[b.key];
//                   });

//         // Filter transaction and compute subtree utilities
//         size_t new_start = filter_transaction(db, i, twu, minutil);

//         // rename items from new start to end to their rank
//         for (size_t j = new_start; j < end; j++)
//         {
//             db.compressed_spare_row_db[j].key = item_rank[db.compressed_spare_row_db[j].key];
//         }

//         compute_subtree_utilities(db, new_start, end, subtree_utility);

//         // Mark empty transactions for removal
//         if (db.csr_transaction_end[i] == db.csr_transaction_start[i])
//         {
//             indices_to_remove.push_back(i);
//         }
//     }

//     // Remove empty transactions
//     remove_empty_transactions(db, indices_to_remove);

//     #ifdef DEBUG
//         std::cout << "Subtree Utility: " << std::endl;
//         for (const auto &item : subtree_utility)
//         {
//             std::cout << item.first << ":" << item.second << std::endl;
//         }

//         std::cout << "Item Rank: " << std::endl;
//         for (const auto &item : item_rank)
//         {
//             std::cout << item.first << ":" << item.second << std::endl;
//         }

//         std::cout << "Database: " << db.db_size << " transactions" << std::endl;
//         for (size_t i = 0; i < db.db_size; i++)
//         {
//             for (size_t j = db.csr_transaction_start[i]; j < db.csr_transaction_end[i]; j++)
//             {
//                 std::cout << db.compressed_spare_row_db[j].key << ":" << db.compressed_spare_row_db[j].value << " ";
//             }
//             std::cout << std::endl;
//         }
//     #endif

//     r.build_time = std::chrono::high_resolution_clock::now();



//     // return 
//     // return subtree_utility, item_rank;
//     return std::make_tuple(std::move(subtree_utility), std::move(item_rank));
// }
