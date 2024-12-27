#include "ImageProcess.h"
#include "ModelRunner.h"
#include <opencv2/opencv.hpp>
#include <iostream>
// model output Shape of outputs[0]: (4, 84, 8400)
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./yolo_inference <model_path.onnx> <image_path>" << std::endl;
        return -1;
    }

    ModelRunner modelRunner(argv[1]);

    // 获取图片列表
    std::vector<std::string> imagePaths;
    ImageProcessor::getImagesFromDirectory(argv[2], imagePaths);
    if (imagePaths.empty()) {
        std::cerr << "No images found in the directory." << std::endl;
        return -1;
    }

    // 加载并预处理图片
    std::vector<cv::Mat> inputImages;
    for (const auto& imagePath : imagePaths) {
        cv::Mat img = cv::imread(imagePath);
        if (!img.empty()) {
            inputImages.push_back(img);
        }
    }

    std::vector<float> batchTensor;
    ImageProcessor::preprocessImages(inputImages, batchTensor, 640, 640);

    // 批量推理
    std::vector<std::vector<float>> outputs;
    modelRunner.runInferenceBatch(batchTensor, inputImages.size(), outputs);

    // 后处理并保存结果
    modelRunner.postprocessBatch(outputs, inputImages, cv::Size(640, 640));
    for (size_t i = 0; i < inputImages.size(); ++i) {
        cv::imwrite("/mnt/f/dataset/miniDataset/result_" + std::to_string(i) + ".jpg", inputImages[i]);
    }

    return 0;
}