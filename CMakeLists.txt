cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}) 
set(TrtPath "/usr/local/TensorRT-8.6.1.6")
set(CudaPath "/usr/local/cuda/")

find_package(OpenCV 4 REQUIRED)

# 收集所有模块的源文件
file(GLOB CORE_SRC "src/core/*.cpp")
file(GLOB MODEL_SRC "src/models/*.cpp") 
file(GLOB BACKEND_SRC "src/backends/*.cpp")
file(GLOB SRC "src/*.cpp")

include_directories(
    ${TrtPath}/include/
    ${CudaPath}/targets/x86_64-linux/include/
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/include/models
    /usr/lib/jvm/jdk-17-oracle-x64/include
    /usr/lib/jvm/jdk-17-oracle-x64/include/linux
)

link_directories(
    /usr/local/lib/
    ${TrtPath}/lib/
    ${CudaPath}/lib64/
)

# add_executable(yolov11_singlePicture
#     ${CORE_SRC}
#     ${MODEL_SRC}
#     ${BACKEND_SRC}
#     ${SRC}
#     examples/yolov11_singlePicture.cpp)

add_library(modelprocessor SHARED
    ${CORE_SRC}
    ${MODEL_SRC}
    ${BACKEND_SRC}
    ${SRC}
    # java/ModelProcessor.cpp
    java/com_jiayang_so_ModelProcessor.cpp
)
set(LIBRARY_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/java)
set_target_properties(modelprocessor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_PATH})
    
target_link_libraries(modelprocessor
    nvinfer
    cudart
    ${OpenCV_LIBRARIES}
    dl
)
