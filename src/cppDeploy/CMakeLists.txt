cmake_minimum_required(VERSION 3.10)
project(YOLOv8_ONNX)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime include and library directories
set(ONNXRUNTIME_DIR /root/joeCodingTools/onnxruntime-linux-x64-gpu-1.18.1)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add util
add_subdirectory(util)

# add main cpp-file
add_executable(yolo_inference onnxDeploy.cpp)
target_link_libraries(yolo_inference ${OpenCV_LIBS} onnxruntime)
