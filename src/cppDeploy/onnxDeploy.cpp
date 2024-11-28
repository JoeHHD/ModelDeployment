// GPU 版本头文件
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem> // C++17 标准
#include <iostream>
#include <vector>
#include <string>

using namespace std;
namespace fs = std::filesystem;
// 等效于 std::string::ends_with，c++20标准
bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
// 读取多张不固定图片的名称
void getImagesFromDirectory(const std::string& folder_path, std::vector<std::string>& image_paths) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            // 检查扩展名是否为图片
            if (ends_with(file_path, ".jpg") || ends_with(file_path, ".png") || ends_with(file_path, ".jpeg")) {
                image_paths.push_back(file_path);
            }
        }
    }
}
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

        if (confidence > 0.90) {
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
    std::string model_path = argv[1];
    Ort::Session session(env, model_path.c_str(), session_options);

	// 创建ort的分配器Allocator with default option
	Ort::AllocatorWithDefaultOptions allocator;
    
    // input shape(s)=(b,c,h,w)
	// get input name ------new version, >= 1.18
    size_t num_inputs = session.GetInputCount();
    std::vector<std::string> input_names(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
	for(size_t i = 0; i < num_inputs; ++i){
		auto input_type_info = session.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::string input_name_str(session.GetInputNameAllocated(i, allocator).get());
        input_names[i] = input_name_str;
		input_shapes[i] = input_tensor_info.GetShape();
        std::cout << "Input name [" << i << "]: " << input_names[i] << std::endl;
        std::cout << "Input shape [" << i << "]: ";
        // 这一行代码打印如果是正常的输出，那模型就是没问题的
        // std::cout << session.GetInputNameAllocated(i,allocator).get() << std::endl;
        for (auto dim : input_shapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
	}

	// get output name ------new version, >=1.18
	size_t num_outputs = session.GetOutputCount();
	std::vector<std::string> output_names(num_outputs);
	std::vector<std::vector<int64_t>> output_shapes(num_outputs);
	for(size_t i = 0; i < num_outputs; i++){
        std::string output_name_str(session.GetOutputNameAllocated(i, allocator).get()); 
		auto output_type_info = session.GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_names[i] = output_name_str;
		output_shapes[i] = output_tensor_info.GetShape();
		std::cout << "Output name [" << i << "]: " << output_names[i] << std::endl;
        std::cout << "Output shape [" << i << "]: ";
        // 这一行代码打印如果是正常的输出，那模型就是没问题的
        // std::cout << session.GetInputNameAllocated(i,allocator).get() << std::endl;
        for (auto dim : output_shapes[i]) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
	}

    // Preprocess image
    std::string folder_path = argv[2];
    // std::vector<std::string> image_paths;
    std::string image_paths = folder_path + "000000000139.jpg";
	size_t input_height = input_shapes[0][2];
	size_t input_width = input_shapes[0][3];
    cv::Mat processed_img;
    // getImagesFromDirectory(folder_path, image_paths);
    // 处理单张图片，处理多张下面逻辑都要改，懒得改 T_T
    cv::Mat image = cv::imread(image_paths);
    if(image.empty()){
        std::cout<<"error: image empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    }
    preprocess(image, processed_img, input_height, input_width);
    // preprocessImages(image_paths, processed_img, input_height, input_width);

    // Prepare input tensor
    size_t input_tensor_size = input_width * input_height * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    memcpy(input_tensor_values.data(), processed_img.data, input_tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, 
        input_shapes[0].data(), input_shapes[0].size() //std::vector<int64_t> input_shapes[0]
    );

    // Run inference
    // GetInputNameAllocated返回的是临时对象，不能直接&取地址
    const char* input_name = input_names[0].c_str(); 
    const char* output_name = output_names[0].c_str();
    auto output_tensors = session.Run(
			Ort::RunOptions{nullptr},
			&input_name, &input_tensor, 1,
			&output_name,
			num_outputs);
    auto output = output_tensors[0].GetTensorMutableData<float>();

    // Postprocess output
    cv::Size input_size(input_width, input_height);
    postprocess(vector<float>(output, output + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()), image, input_size);

    // Save the result in local
    std::string imgSavePath = "/home/joe/project/pth2onnx/images/processed_imgs/processed_img000001.jpg";
    bool isSaved = cv::imwrite(imgSavePath, image);
    if (isSaved) {
        std::cout << "Image saved successfully." << std::endl;
    } else {
        std::cout << "Failed to save image." << std::endl;
    }
    return 0;
}
