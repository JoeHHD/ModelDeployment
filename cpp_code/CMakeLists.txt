cmake_minimum_required(VERSION 3.10)
project(YOLOInference)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 include 目录
include_directories(include)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 自动收集 src/ 目录下的所有源文件
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# 添加 main.cc 文件
set(MAIN_FILE main.cpp)

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT "/root/joeCodingTools/onnxruntime-linux-x64-gpu-1.18.1")

# 包含 ONNX Runtime 的头文件
include_directories(${ONNXRUNTIME_ROOT}/include)

# 链接 ONNX Runtime 库
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT}/lib")
endif()

# 定义目标可执行文件
add_executable(YOLOInference ${MAIN_FILE} ${SRC_FILES})

# 链接 OpenCV 和 ONNX Runtime
target_link_libraries(YOLOInference PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})

# 设置输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
