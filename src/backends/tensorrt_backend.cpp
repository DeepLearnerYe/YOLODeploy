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
    : IInferBackend(modelLoadOpt), runtime_(nullptr), engine_(nullptr), context_(nullptr), stream_(nullptr)
{
}

TensorRTBackend::~TensorRTBackend()
{
    std::lock_guard<std::mutex> lock(mutex_);
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
    unsigned char* modelPtr = nullptr;
    unsigned int modelLen;
    OpenLibrary(modelLoadOpt_.modelPath, modelPtr, modelLen, labels_);
    unsigned char *serialized_engine = new u_char[modelLen];
    std::memcpy(serialized_engine, modelPtr, modelLen);

    engine_ = runtime_->deserializeCudaEngine(serialized_engine, modelLen);
    assert(engine_);
    context_ = engine_->createExecutionContext();
    assert(context_);
    cudaStreamCreate(&stream_);

    auto numTensors = engine_->getNbIOTensors();
    for(int i = 0; i < numTensors; ++i)
    {
        auto tensorName = engine_->getIOTensorName(i);
        if (nvinfer1::TensorIOMode::kINPUT == engine_->getTensorIOMode(tensorName))
        {
            inputDims_ = engine_->getTensorShape(tensorName);
        }else if(nvinfer1::TensorIOMode::kOUTPUT == engine_->getTensorIOMode(tensorName))
        {
            outputDims_ = engine_->getTensorShape(tensorName);
        }else
        {
            std::cout << "get tensor name fail " << std::endl;
            return -1;
        }
    }

    cudaMalloc(&deviceBuffers_[0], modelLoadOpt_.batch * 3 * inputDims_.d[2] * inputDims_.d[3] * sizeof(float));
    cudaMalloc(&deviceBuffers_[1], modelLoadOpt_.batch * outputDims_.d[1] * outputDims_.d[2] * sizeof(float));
    hostBuffer_ = new float[modelLoadOpt_.batch * outputDims_.d[1] * outputDims_.d[2]];
    output_.reserve(modelLoadOpt_.batch * outputDims_.d[1] * outputDims_.d[2]);

    delete[] serialized_engine;
    return 0;
}

int TensorRTBackend::SetInput(void *data, size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t expectedSize = modelLoadOpt_.batch * 3 * inputDims_.d[2] * inputDims_.d[3] * sizeof(float);
    if(size * sizeof(float) != expectedSize)
    {
        std::cout << "Invalid input size: expected " << expectedSize << ", got " << size << std::endl;
        return -1;
    }
    cudaMemcpyAsync(deviceBuffers_[0], data, expectedSize, cudaMemcpyHostToDevice, stream_);
    return 0;
}

int TensorRTBackend::Infer()
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto status = context_->enqueueV2(deviceBuffers_, stream_, nullptr);
    cudaStreamSynchronize(stream_);
    return status;
}

const std::vector<float> &TensorRTBackend::GetOutput()
{
    std::lock_guard<std::mutex> lock(mutex_);
    cudaMemcpyAsync(hostBuffer_, deviceBuffers_[1], modelLoadOpt_.batch * outputDims_.d[1] * outputDims_.d[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    output_.assign(hostBuffer_, hostBuffer_ + modelLoadOpt_.batch * outputDims_.d[1] * outputDims_.d[2]);
    return output_;
}

Shape TensorRTBackend::GetOutputShape()
{
    Shape output{{outputDims_.d[0], outputDims_.d[1], outputDims_.d[2]}};
    return output;
}