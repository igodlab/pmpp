#include "utils.hpp"
#include <stdexcept>
#include <iostream>

cv::Mat loadImage(const std::string& path, int readMode) {
    cv::Mat img = cv::imread(path, readMode);
    if (img.empty())
        throw std::runtime_error("Could not load image: " + path);
    if (!img.isContinuous())
        img = img.clone();
    return img;
}

void saveImage(const std::string& outputPath, const cv::Mat& image) {
    if (image.empty())
        throw std::runtime_error("Cannot save empty image to: " + outputPath);
    bool success = cv::imwrite(outputPath, image);
    if (!success)
        throw std::runtime_error("Failed to write image to: " + outputPath);
    std::cout << "Image saved to: " << outputPath << "\n";
}
