#include "ImageProcess.h"
#include <iostream>

bool ImageProcessor::endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void ImageProcessor::getImagesFromDirectory(const std::string& folderPath, std::vector<std::string>& imagePaths) {
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            if (endsWith(filePath, ".jpg") || endsWith(filePath, ".png") || endsWith(filePath, ".jpeg")) {
                imagePaths.push_back(filePath);
            }
        }
    }
}

void ImageProcessor::preprocess(const cv::Mat& image, cv::Mat& processedImg, int inputHeight, int inputWidth) {
    cv::Mat resized;
    float scale = std::min(inputWidth / (float)image.cols, inputHeight / (float)image.rows);
    int nw = (int)(image.cols * scale);
    int nh = (int)(image.rows * scale);
    cv::resize(image, resized, cv::Size(nw, nh));

    processedImg = cv::Mat::zeros(cv::Size(inputWidth, inputHeight), CV_8UC3);
    resized.copyTo(processedImg(cv::Rect(0, 0, nw, nh)));
    processedImg.convertTo(processedImg, CV_32F, 1.0 / 255);
}

void ImageProcessor::preprocessImages(const std::vector<cv::Mat>& images, 
                                            std::vector<float>& batchTensor, 
                                            int inputHeight, int inputWidth) {
    int batchSize = images.size();
    int inputChannels = 3; // 假设 RGB 格式
    batchTensor.resize(batchSize * inputChannels * inputHeight * inputWidth);

    for (int i = 0; i < batchSize; ++i) {
        cv::Mat resized, processedImg;
        float scale = std::min(inputWidth / (float)images[i].cols, 
                                inputHeight / (float)images[i].rows);
        int nw = (int)(images[i].cols * scale);
        int nh = (int)(images[i].rows * scale);

        // Resize and pad
        cv::resize(images[i], resized, cv::Size(nw, nh));
        processedImg = cv::Mat::zeros(cv::Size(inputWidth, inputHeight), CV_8UC3);
        resized.copyTo(processedImg(cv::Rect(0, 0, nw, nh)));

        // Normalize
        processedImg.convertTo(processedImg, CV_32F, 1.0 / 255);

        // Copy data into batch tensor
        float* batchData = batchTensor.data() + i * inputChannels * inputHeight * inputWidth;
        std::memcpy(batchData, processedImg.data, inputChannels * inputHeight * inputWidth * sizeof(float));
    }
}

