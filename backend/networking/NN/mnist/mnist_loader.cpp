#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>

std::vector<std::vector<uint8_t>> loadMNISTImages(const std::string &fileName, int num_images)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "loadMNISTImages - Error: Could not open: " << fileName << std::endl;
        exit(1);
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    std::cout << "File size: " << fileSize << " bytes" << std::endl;
    file.seekg(0, std::ios::beg); // move read pointer back to beginning

    uint32_t magic = 0, n_images = 0, n_rows = 0, n_cols = 0;
    file.read((char *)&magic, sizeof(magic));
    file.read((char *)&n_images, sizeof(n_images));
    file.read((char *)&n_rows, sizeof(n_rows));
    file.read((char *)&n_cols, sizeof(n_cols));

    // Convert from big-endian to host byte order
    magic = __builtin_bswap32(magic);
    n_images = __builtin_bswap32(n_images);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    if (magic != 0x803)
    {
        std::cerr << "Invalid MNIST image file!" << std::endl;
        exit(1);
    }

    num_images = std::min(num_images, static_cast<int>(n_images));

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(n_rows * n_cols));

    for (int i = 0; i < num_images; i++)
    {
        std::vector<uint8_t> &image = images[i];
        file.read(reinterpret_cast<char *>(image.data()), n_rows * n_cols);

        if (!file)
        {
            std::cerr << "Error reading image " << i << std::endl;
            exit(1);
        }
    }
    file.close();

    std::cout << "Loaded " << num_images << " images of size " << n_rows << "x" << n_cols << std::endl;
    return images;
}

std::vector<uint8_t> loadMNISTLabels(const std::string &fileName, int num_images)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open " << fileName << std::endl;
        exit(1);
    }

    file.ignore(8); // Skip header
    std::vector<uint8_t> labels(num_images);
    file.read(reinterpret_cast<char *>(labels.data()), num_images);

    file.close();
    return labels;
}
