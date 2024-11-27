// GPU 版本头文件
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;

// 一次预处理一张图片
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
// 一次预处理多张图片
void preprocessImages(const std::vector<std::string>& image_paths, 
					  std::vector<cv::Mat>& processed_images, 
					  int input_height, 
					  int input_width){
	for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            continue;
        }

        cv::Mat processed_img;
        preprocess(image, processed_img, input_height, input_width);

        // 保存或进一步处理
        // std::cout << "Processed image: " << image_path << std::endl;
    }
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

	// init onnxruntime
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
	Ort::SessionOptions session_options;
	// init CUDA providers
	OrtCUDAProviderOptions options;
	options.device_id = 0;
	options.arena_extend_strategy = 0;
	options.gpu_mem_limit = SIZE_MAX;
	options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	options.do_copy_in_default_stream = 1;
	session_options.AppendExecutionProvider_CUDA(options);

	// load onnx model
    Ort::Session session(env, model_path.c_str(), session_options);

	// 创建ort的分配器Allocator with default option
	Ort::AllocatorWithDefaultOptions allocator;

	// get input name ------new version, >= 1.18
    size_t num_inputs = session.GetInputCount();
    std::vector<const char*> input_names(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);//TODO: why?
	for(size_t i = 0; i < num_inputs; ++i){
		input_names[i] = session.GetInputNameAllocated(i, allocator).get();
		auto input_type_info = session.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		input_shapes[i] = input_tensor_info.GetShape();
        std::cout << "Input name [" << i << "]: " << input_names[i] << std::endl;
        std::cout << "Input shape [" << i << "]: ";
        for (auto dim : input_shapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
	}

	// get output name ------new version, >=1.18
	size_t num_outputs = session.GetOutputCount();
	std::vector<const char*> output_names(num_outputs);
	std::vector<std::vector<int64_t>> output_shapes(num_outputs);
	for(size_t i = 0; i < num_outputs; i++){
		output_names[i] = session.GetOutputNameAllocated(i, allocator).get();
		auto output_type_info = session.GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		output_shapes[i] = output_tensor_info.GetShape();
		std::cout << "Output name [" << i << "]: " << output_names[i] << std::endl;
        std::cout << "Output shape [" << i << "]: ";
        for (auto dim : output_shapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
	}

    // Preprocess image
	size_t input_height = input_shapes[0][1];
	size_t input_width = input_shapes[0][2];
    cv::Mat image = cv::imread(image_path);
    cv::Mat processed_img;
    preprocess(image, processed_img, input_height, input_width);

    // Prepare input tensor
    size_t input_tensor_size = input_width * input_height * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    memcpy(input_tensor_values.data(), processed_img.data, input_tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shapes[0].data(), input_shapes[0].size());

    // Run inference
    auto output_tensors = session.Run(
			Ort::RunOptions{nullptr}, 
			&session.GetInputNameAllocated(0, allocator).get(), &input_tensor, 1, 
			&session.GetOutputName(0), 
			1);
    auto output = output_tensors[0].GetTensorMutableData<float>();

    // Postprocess output
    cv::Size input_size(input_width, input_height);
    postprocess(vector<float>(output, output + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()), image, input_size);

    // Display the result
    cv::imshow("Result", image);
    cv::waitKey(0);

    return 0;
}
