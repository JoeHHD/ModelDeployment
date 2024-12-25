#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ModelRunner {
public:
    explicit ModelRunner(const std::string& modelPath);
    void runInference(const cv::Mat& inputImage, std::vector<float>& output);
    void runInferenceBatch(std::vector<float>& batchTensor, size_t batchSize, 
                            std::vector<std::vector<float>>& outputs);
    void postprocess(const std::vector<float>& output, const cv::Mat& inputImage, 
                        const cv::Size& inputSize);
    void postprocessBatch(const std::vector<std::vector<float>>& outputs, 
                            const std::vector<cv::Mat>& inputImages, const cv::Size& inputSize);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> inputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::string> outputNames;
    std::vector<std::vector<int64_t>> outputShapes;
};

#endif // MODEL_RUNNER_H
