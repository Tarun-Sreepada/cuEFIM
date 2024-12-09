#include "file.hpp"
// #include "structure.cuh"

ssize_t file_size(const std::string &file_path)
{
    struct stat st;
    if (stat(file_path.c_str(), &st) != 0)
    {
        return -1;
    }
    return st.st_size;
}

raw_file read_file_cpu(results &r, Config::Params &p)
{

    // check if the file exists
    if (access(p.input_file.c_str(), F_OK) == -1)
    {
        std::cerr << "Error: File does not exist" << std::endl;
        exit(1);
    }

    ssize_t file_size_bytes = file_size(p.input_file);

    if (file_size_bytes == -1)
    {
        std::cerr << "Error: Unable to get file size" << std::endl;
        exit(1);
    }
    else if (file_size_bytes == 0)
    {
        std::cerr << "Error: File is empty" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "File size: " << file_size_bytes << " bytes" << std::endl;
    }

    raw_file rf;
    rf.size_bytes = file_size_bytes;
    rf.data = (char *)malloc(rf.size_bytes);

    if (rf.data == nullptr)
    {
        std::cerr << "Error: Unable to allocate memory for file" << std::endl;
        exit(1);
    }

    rf.fd = open(p.input_file.c_str(), O_RDONLY);
    if (rf.fd == -1)
    {
        std::cerr << "Error: Unable to open file" << std::endl;
        exit(1);
    }

    #ifdef __linux__
        // Set the file to be read ahead
        if (posix_fadvise(rf.fd, 0, 0, POSIX_FADV_WILLNEED) != 0)
        {
            std::cerr << "Error: Unable to set file to be read ahead" << std::endl;
            close(rf.fd);
            free(rf.data);
            exit(1);
        }
    #endif

    struct io_uring ring;
    if (io_uring_queue_init(p.queue_depth, &ring, 0) < 0)
    {
        std::cerr << "Error: Unable to initialize io_uring" << std::endl;
        close(rf.fd);
        free(rf.data);
        exit(1);
    }

    // Submit and process read requests
    struct io_uring_cqe *cqes[p.queue_depth];

    std::unordered_map<uint32_t, uint32_t> twu;

    while (rf.submitted_bytes < rf.size_bytes || !rf.retrieved_indices.empty())
    {
        // Submit as many requests as possible within queue depth limits
        uint32_t free_submission_entry = io_uring_sq_space_left(&ring);

        for (size_t i = 0; i < free_submission_entry && rf.submitted_bytes < rf.size_bytes; i++)
        {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe)
            {
                std::cerr << "Error: Unable to get submission queue entry" << std::endl;
                break;
            }

            size_t bytes_to_read = std::min(p.page_size, rf.size_bytes - rf.submitted_bytes);
            io_uring_prep_read(sqe, rf.fd, rf.data + rf.submitted_bytes, bytes_to_read, rf.submitted_bytes);
            sqe->user_data = rf.submitted_bytes;
            rf.submitted_bytes += bytes_to_read;
        }

        if (io_uring_submit(&ring) < 0)
        {
            std::cerr << "Error: Unable to submit read requests" << std::endl;
            break;
        }

        // Process completed requests
        int completed = io_uring_peek_batch_cqe(&ring, cqes, p.queue_depth);
        while (completed > 0)
        {
            for (int i = 0; i < completed; i++)
            {
                struct io_uring_cqe *cqe = cqes[i];
                if (cqe->res < 0)
                {
                    std::cerr << "Error: Read request failed" << std::endl;
                    break;
                }

                size_t retrieved_bytes = cqe->res;
                size_t user_data = cqe->user_data;

                rf.retrieved_indices.emplace_back(user_data, retrieved_bytes);

                io_uring_cqe_seen(&ring, cqe);
            }
            completed = io_uring_peek_batch_cqe(&ring, cqes, p.queue_depth);
        }

        if (completed < 0)
        {
            std::cerr << "Error: Unable to peek completion queue entries" << std::endl;
            break;
        }

        // Maintain retrieved_indices sorted order
        std::sort(rf.retrieved_indices.begin(), rf.retrieved_indices.end());

        // Process sequential retrieved indices
        while (!rf.retrieved_indices.empty() && rf.retrieved_indices.front().first == rf.retrieved_bytes)
        {
            auto [offset, length] = rf.retrieved_indices.front();
            rf.retrieved_indices.erase(rf.retrieved_indices.begin());
            rf.retrieved_bytes += length;
        }
    }

    io_uring_queue_exit(&ring);

    r.record_memory_usage("File Read");
    close(rf.fd);

    return std::move(rf);
}
