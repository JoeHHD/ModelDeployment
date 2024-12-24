#include "ImageProcessor.h"
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

void ImageProcessor::preprocessImagesModelDeployment/src/cppDeploy/src ModelDeployment/src/cppDeploy/src/ImageProcess.cc ModelDeployment/src/cppDeploy/src/ModelRunner.cc ModelDeployment/src/cppDeploy/src/nms.cpp(const std::vector<std::string>& imagePaths, std::vector<cv::Mat>& processedImages, int inputHeight, int inputWidth) {
    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        cv::Mat processedImg;
        preprocess(image, processedImg, inputHeight, inputWidth);
        processedImages.push_back(processedImg);
    }
}
