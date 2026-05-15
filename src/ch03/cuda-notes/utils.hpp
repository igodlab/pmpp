#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat loadImage(const std::string& path, int readMode = cv::IMREAD_COLOR);

void saveImage(const std::string& outputPath, const cv::Mat& image);
