#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

void preprocess(const cv::Mat& image, cv::Mat& processed_img, int input_height, int input_width) {
    cv::Mat resized;
    // Resize and pad the image
    float scale = min(input_width / (float)image.cols, input_height / (float)image.rows);
    int nw = (int)(image.cols * scale);
    int nh = (int)(image.rows * scale);
    cv::resize(image, resized, cv::Size(nw, nh));

    // Padding
    processed_img = cv::Mat::zeros(cv::Size(input_width, input_height), CV_8UC3);
    resized.copyTo(processed_img(cv::Rect(0, 0, nw, nh)));

    // Normalize
    processed_img.convertTo(processed_img, CV_32F, 1.0 / 255);
}

void postprocess(const std::vector<float>& output, const cv::Mat& input_img, const cv::Size& input_size) {
    // Output format: [x1, y1, x2, y2, confidence, class_id]
    int num_detections = output.size() / 6; // Assuming 6 elements per detection
    for (int i = 0; i < num_detections; ++i) {
        float x1 = output[i * 6];
        float y1 = output[i * 6 + 1];
        float x2 = output[i * 6 + 2];
        float y2 = output[i * 6 + 3];
        float confidence = output[i * 6 + 4];
        int class_id = (int)output[i * 6 + 5];

        if (confidence > 0.25) {
            // Draw bounding box
            cv::rectangle(input_img, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 0), 2);
            // Display class and confidence
            string label = "Class " + to_string(class_id) + ": " + to_string(confidence);
            cv::putText(input_img, label, cv::Point((int)x1, (int)y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}

int main(int argc, char** argv) {
    // Check arguments
    if (argc < 3) {
        cerr << "Usage: ./yolo_inference <model_path.onnx> <image_path>" << endl;
        return -1;
    }

    string model_path = argv[1];
    string image_path = argv[2];

    // Load the ONNX model
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    // Input metadata
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int input_height = input_shape[2];
    int input_width = input_shape[3];

    // Preprocess image
    cv::Mat image = cv::imread(image_path);
    cv::Mat processed_img;
    preprocess(image, processed_img, input_height, input_width);

    // Prepare input tensor
    size_t input_tensor_size = input_width * input_height * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    memcpy(input_tensor_values.data(), processed_img.data, input_tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &session.GetInputName(0), &input_tensor, 1, &session.GetOutputName(0), 1);
    auto output = output_tensors[0].GetTensorMutableData<float>();

    // Postprocess output
    cv::Size input_size(input_width, input_height);
    postprocess(vector<float>(output, output + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()), image, input_size);

    // Display the result
    cv::imshow("Result", image);
    cv::waitKey(0);

    return 0;
}
