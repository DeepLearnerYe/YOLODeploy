cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}) 
set(TrtPath "/usr/local//TensorRT-8.6.1.6")
set(CudaPath "/usr/local/cuda/")

find_package(OpenCV 4 REQUIRED)

include_directories(
    ${TrtPath}/include/
    ${CudaPath}/targets/x86_64-linux/include/
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/
)

link_directories(
    /usr/local/lib/
    ${TrtPath}/lib/
    ${CudaPath}lib64/
   
)

set(sourceList API.cpp Model.cpp Base64.cpp Log.cpp)

# 核心模块
add_library(core 
    include/core/base_model.hpp
    src/core/base_model.cpp
)

# TensorRT后端
add_library(tensorrt_backend
    include/backends/tensorrt/engine.hpp
    src/backends/tensorrt/engine.cpp
)
target_link_libraries(tensorrt_backend PRIVATE core)

# YOLOv5模型
add_library(yolov5
    include/models/detection/yolov5.hpp
    src/models/detection/yolov5.cpp
)
target_link_libraries(yolov5 PRIVATE core tensorrt_backend)
