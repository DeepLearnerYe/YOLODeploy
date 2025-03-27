#ifndef TENSORRT_BACKEND_HPP
#define TENSORRT_BACKEND_HPP

#include <NvInfer.h>
#include "backend.hpp"

class TensorRTBackend:public IInferBackend
{
public:
    explicit TensorRTBackend(const ModelLoadOpt& modelLoadOpt);
    ~TensorRTBackend() override;

    int Initialize() override;
    int SetInput(void* data, size_t size) override;
    int Infer() override;
    const std::vector<float>& GetOutput() override;
    Shape GetOutputShape() override;
private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    nvinfer1::Dims inputDims_;
    nvinfer1::Dims outputDims_;
    size_t outputSize_;
    const char* inputTensorName_;
    const char* outputTensorName_;
};

#endif