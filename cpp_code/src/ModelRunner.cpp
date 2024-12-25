#include "ModelRunner.h"
#include "nms.h"
#include <iostream>
#include <cstring> // For memcpy

ModelRunner::ModelRunner(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
      session(env, modelPath.c_str(), Ort::SessionOptions{}) {
    // 获取输入信息
    size_t numInputs = session.GetInputCount();
    inputNames.resize(numInputs);
    inputShapes.resize(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        auto inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputNames[i] = session.GetInputNameAllocated(i, allocator).get();
        inputShapes[i] = inputTensorInfo.GetShape();
        std::cout << "Input name [" << i << "]: " << inputNames[i] << std::endl;
        std::cout << "Input shape [" << i << "]: ";
        for (auto dim : inputShapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    // 获取输出信息
    size_t numOutputs = session.GetOutputCount();
    outputNames.resize(numOutputs);
    outputShapes.resize(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        auto outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputNames[i] = session.GetOutputNameAllocated(i, allocator).get();
        outputShapes[i] = outputTensorInfo.GetShape();
        std::cout << "Output name [" << i << "]: " << outputNames[i] << std::endl;
        std::cout << "Output shape [" << i << "]: ";
        for (auto dim : outputShapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}
// 推理单张图片
void ModelRunner::runInference(const cv::Mat& inputImage, std::vector<float>& output) {
    if (inputShapes.empty() || inputShapes[0].size() < 4) {
        throw std::runtime_error("Invalid input shape.");
    }

    int64_t inputHeight = inputShapes[0][2];
    int64_t inputWidth = inputShapes[0][3];
    int64_t inputChannels = inputShapes[0][1];

    if (inputImage.rows != inputHeight || inputImage.cols != inputWidth) {
        throw std::runtime_error("Input image size does not match the model's expected input size.");
    }

    size_t inputTensorSize = inputHeight * inputWidth * inputChannels;
    std::vector<float> inputTensorValues(inputTensorSize);
    std::memcpy(inputTensorValues.data(), inputImage.data, inputTensorSize * sizeof(float));

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputShapes[0].data(), inputShapes[0].size());

    const char* inputName = inputNames[0].c_str();
    const char* outputName = outputNames[0].c_str();

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        &inputName, &inputTensor, 1,
        &outputName, outputNames.size());

    size_t outputTensorSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    auto* outputData = outputTensors[0].GetTensorMutableData<float>();
    output.assign(outputData, outputData + outputTensorSize);
}
// 推理多张图片
void ModelRunner::runInferenceBatch(std::vector<float>& batchTensor, 
                                size_t batchSize, 
                                std::vector<std::vector<float>>& outputs) {
    // 更新输入形状以支持批量维度
    std::vector<int64_t> batchInputShape = inputShapes[0];
    batchInputShape[0] = batchSize;

    // 创建输入 Tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, batchTensor.data(), batchTensor.size(),
        batchInputShape.data(), batchInputShape.size());

    // 运行推理
    const char* inputName = inputNames[0].c_str();
    const char* outputName = outputNames[0].c_str();
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        &inputName, &inputTensor, 1,
        &outputName, outputNames.size());

    // 提取输出数据
    size_t outputTensorSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    size_t perBatchOutputSize = outputTensorSize / batchSize;

    // 将输出分离到每张图片
    outputs.resize(batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        outputs[i].assign(outputData + i * perBatchOutputSize, outputData + (i + 1) * perBatchOutputSize);
    }
}

void ModelRunner::postprocess(const std::vector<float>& output, const cv::Mat& inputImage, const cv::Size& inputSize) {
    int numDetections = output.size() / 6;
    std::vector<Detection> detections;

    // 将输出解析为检测结果
    for (int i = 0; i < numDetections; ++i) {
        float x = output[i * 6];
        float y = output[i * 6 + 1];
        float w = output[i * 6 + 2];
        float h = output[i * 6 + 3];
        float confidence = output[i * 6 + 4];
        int classId = static_cast<int>(output[i * 6 + 5]);

        if (confidence > 0.8) {  // 置信度阈值
            detections.push_back({{x, y, w, h}, confidence, classId});
        }
    }

    // 执行非极大值抑制
    nms nmsProcessor;
    std::vector<Detection> filteredDetections = nmsProcessor.nonMaxSuppression(detections, 0.3);
    // 限制检测框数量
    int maxBoxes = 3;
    if (filteredDetections.size() > maxBoxes) {
        filteredDetections.resize(maxBoxes);
    }
    // 绘制检测框
    for (const auto& detection : filteredDetections) {
        const auto& box = detection.box;
        float x1 = box.x - box.w / 2;
        float y1 = box.y - box.h / 2;
        float x2 = box.x + box.w / 2;
        float y2 = box.h + box.h / 2;

        cv::rectangle(inputImage, cv::Point((int)x1, (int)y1), 
                        cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 0), 2);
        std::string label = "Class: " + std::to_string(detection.class_id) + " Confidence: " + 
                                std::to_string(detection.confidence);
        cv::putText(inputImage, label, cv::Point((int)x1, (int)y1 - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}
void ModelRunner::postprocessBatch(const std::vector<std::vector<float>>& outputs, 
                                    const std::vector<cv::Mat>& inputImages, const cv::Size& inputSize) {
    for (size_t i = 0; i < outputs.size(); ++i) {
        postprocess(outputs[i], inputImages[i], inputSize);
    }
}