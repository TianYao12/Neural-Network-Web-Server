#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>
#include <cstdint>

std::vector<std::vector<uint8_t>> loadMNISTImages(const std::string &fileName, int num_images);
std::vector<uint8_t> loadMNISTLabels(const std::string &fileName, int num_images);

#endif
