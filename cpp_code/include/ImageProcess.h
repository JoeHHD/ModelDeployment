#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

class ImageProcessor {
public:
    static bool endsWith(const std::string& str, const std::string& suffix);
    static void getImagesFromDirectory(const std::string& folderPath, std::vector<std::string>& imagePaths);
    static void preprocess(const cv::Mat& image, cv::Mat& processedImg, int inputHeight, int inputWidth);
    static void preprocessImages(const std::vector<cv::Mat>& images, std::vector<float>& batchTensor, 
                                                        int inputHeight, int inputWidth);
};

#endif // IMAGE_PROCESSOR_H
