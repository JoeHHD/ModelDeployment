#include "ModelRunner.h"
#include <iostream>

ModelRunner::ModelRunner(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
      session(env, modelPath.c_str(), Ort::SessionOptions{}) {
    size_t numInputs = session.GetInputCount();
    inputNames.resize(numInputs);
    inputShapes.resize(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        inputNames[i] = session.GetInputNameAllocated(i, allocator).get();
        auto inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShapes[i] = inputTensorInfo.GetShape();
    }

    size_t numOutputs = session.GetOutputCount();
    outputNames.resize(numOutputs);
    outputShapes.resize(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        outputNames[i] = session.GetOutputNameAllocated(i, allocator).get();
        auto outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShapes[i] = outputTensorInfo.GetShape();
    }
}

void ModelRunner::runInference(const cv::Mat& inputImage, std::vector<float>& output) {
    size_t inputTensorSize = inputImage.total() * inputImage.elemSize();
    std::vector<float> inputTensorValues(inputTensorSize);
    memcpy(inputTensorValues.data(), inputImage.data, inputTensorSize);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputShapes[0].data(), inputShapes[0].size());

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(), &inputTensor, 1,
        outputNames.data(), outputNames.size());

    auto* outputData = outputTensors[0].GetTensorMutableData<float>();
    output.assign(outputData, outputData + outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
}

void ModelRunner::postprocess(const std::vector<float>& output, const cv::Mat& inputImage, const cv::Size& inputSize) {
    int numDetections = output.size() / 6;
    for (int i = 0; i < numDetections; ++i) {
        float x1 = output[i * 6];
        float y1 = output[i * 6 + 1];
        float x2 = output[i * 6 + 2];
        float y2 = output[i * 6 + 3];
        float confidence = output[i * 6 + 4];
        int classId = (int)output[i * 6 + 5];

        if (confidence > 0.90) {
            cv::rectangle(inputImage, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 0), 2);
            std::string label = "Class " + std::to_string(classId) + ": " + std::to_string(confidence);
            cv::putText(inputImage, label, cv::Point((int)x1, (int)y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}
