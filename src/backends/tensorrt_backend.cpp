#include <cassert>
#include <fstream>
#include <iostream>
#include <cstring>
#include "utils.hpp"
#include "tensorrt_backend.hpp"

// ======================= need to optimize
const char *LogLevelToString[] = {
    "FATAL",
    "ERROR",
    "WARN",
    "INFO",
    "DEBUG"};

void defaultLogcallback(unsigned int level, const char *msg)
{
    std::time_t now = std::time(nullptr);
    std::tm localTime;
    localtime_r(&now, &localTime);
    char timeBuffer[20];
    std::strftime(timeBuffer, 20, "%Y-%m-%d %H:%M:%S", &localTime);
    printf("[%s], %s, %s\n", LogLevelToString[level], timeBuffer, msg);
}

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
        {
            defaultLogcallback(3, msg);
        }
    }
} gLogger;

// ======================= need to optimize

TensorRTBackend::TensorRTBackend(const ModelLoadOpt &modelLoadOpt)
    : IInferBackend(modelLoadOpt), runtime_(nullptr), engine_(nullptr), context_(nullptr), stream_(nullptr), outputSize_(1),
      inputTensorName_(nullptr), outputTensorName_(nullptr)
{
}

TensorRTBackend::~TensorRTBackend()
{
    if (stream_)
    {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    if (context_)
    {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_)
    {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_)
    {
        runtime_->destroy();
        runtime_ = nullptr;
    }
    if (deviceBuffers_[0])
    {
        cudaFree(deviceBuffers_[0]);
        deviceBuffers_[0] = nullptr;
    }
    if (deviceBuffers_[1])
    {
        cudaFree(deviceBuffers_[1]);
        deviceBuffers_[1] = nullptr;
    }
    if (hostBuffer_)
    {
        delete[] hostBuffer_;
        hostBuffer_ = nullptr;
    }
}

int TensorRTBackend::Initialize()
{
    cudaSetDevice(modelLoadOpt_.deviceId);
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_ != nullptr);

    // read model data from "so"
    unsigned char *modelPtr = nullptr;
    unsigned int modelLen;
    OpenLibrary(modelLoadOpt_.modelPath, modelPtr, modelLen, labels_);
    unsigned char *serialized_engine = new u_char[modelLen];
    std::memcpy(serialized_engine, modelPtr, modelLen);

    engine_ = runtime_->deserializeCudaEngine(serialized_engine, modelLen);
    assert(engine_);
    context_ = engine_->createExecutionContext();
    assert(context_);
    cudaStreamCreate(&stream_);

    // get the input and output of model
    auto numTensors = engine_->getNbIOTensors();
    for (int i = 0; i < numTensors; ++i)
    {
        auto tensorName = engine_->getIOTensorName(i);
        if (nvinfer1::TensorIOMode::kINPUT == engine_->getTensorIOMode(tensorName))
        {
            inputDims_ = engine_->getTensorShape(tensorName);
            inputTensorName_ = tensorName;
        }
        else if (nvinfer1::TensorIOMode::kOUTPUT == engine_->getTensorIOMode(tensorName))
        {
            outputDims_ = engine_->getTensorShape(tensorName);
            outputTensorName_ = tensorName;
        }
        else
        {
            std::cout << "get tensor name fail " << std::endl;
            return -1;
        }
    }
    for(int i = 0; i < outputDims_.nbDims; ++i)
    {
        outputSize_ *= outputDims_.d[i];
    }

    // assign memory
    cudaMalloc(&deviceBuffers_[0], modelLoadOpt_.batch * 3 * inputDims_.d[2] * inputDims_.d[3] * sizeof(float));
    cudaMalloc(&deviceBuffers_[1], modelLoadOpt_.batch * outputSize_ * sizeof(float));
    hostBuffer_ = new float[modelLoadOpt_.batch * outputSize_];
    output_.reserve(modelLoadOpt_.batch * outputSize_);
    // if (inputTensorName_ && outputTensorName_)
    // {
    //     context_->setTensorAddress(inputTensorName_, deviceBuffers_[0]);
    //     context_->setTensorAddress(outputTensorName_, deviceBuffers_[1]);
    // }
    delete[] serialized_engine;
    return 0;
}

int TensorRTBackend::SetInput(void *data, size_t size)
{
    size_t expectedSize = modelLoadOpt_.batch * 3 * inputDims_.d[2] * inputDims_.d[3] * sizeof(float);
    if (size * sizeof(float) != expectedSize)
    {
        std::cout << "Invalid input size: expected " << expectedSize << ", got " << size << std::endl;
        return -1;
    }
    cudaMemcpyAsync(deviceBuffers_[0], data, expectedSize, cudaMemcpyHostToDevice, stream_);
    return 0;
}

int TensorRTBackend::Infer()
{
    auto status = context_->enqueueV2(deviceBuffers_, stream_, nullptr);
    // auto status = context_->enqueueV3(stream_);
    // cudaStreamSynchronize(stream_);
    return status;
}

const std::vector<float> &TensorRTBackend::GetOutput()
{
    cudaMemcpyAsync(hostBuffer_, deviceBuffers_[1], modelLoadOpt_.batch * outputSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    output_.assign(hostBuffer_, hostBuffer_ + modelLoadOpt_.batch * outputSize_);
    return output_;
}

Shape TensorRTBackend::GetOutputShape()
{
    Shape outputShape;
    for(int i = 0; i < outputDims_.nbDims; ++i)
    {
        outputShape.dims.push_back(outputDims_.d[i]);
    }
    return outputShape;
}