#include "ImageProcess.h"
#include "ModelRunner.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./yolo_inference <model_path.onnx> <image_path>" << std::endl;
        return -1;
    }

    ModelRunner modelRunner(argv[1]);
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat processedImage;
    ImageProcessor::preprocess(image, processedImage, 640, 640);

    std::vector<float> output;
    modelRunner.runInference(processedImage, output);

    modelRunner.postprocess(output, image, cv::Size(640, 640));
    cv::imwrite("result.jpg", image);

    return 0;
}
