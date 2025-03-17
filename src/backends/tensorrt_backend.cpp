#include <cassert>
#include <fstream>
#include <iostream>
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
    std::ifstream file(modelLoadOpt_.modelPath, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "read " << modelLoadOpt_.modelPath << " error!" << std::endl;
        return -1;
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    engine_ = runtime_->deserializeCudaEngine(serialized_engine, size);
    assert(engine_);
    context_ = engine_->createExecutionContext();
    assert(context_);
    cudaStreamCreate(&stream_);

    cudaMalloc(&deviceBuffers_[0], modelLoadOpt_.batch * 3 * modelLoadOpt_.inputHeight * modelLoadOpt_.inputWidth * sizeof(float));
    cudaMalloc(&deviceBuffers_[1], modelLoadOpt_.batch * modelLoadOpt_.OutputSize * sizeof(float));
    hostBuffer_ = new float[modelLoadOpt_.batch * modelLoadOpt_.OutputSize];
    output_.reserve(modelLoadOpt_.batch * modelLoadOpt_.OutputSize);

    delete[] serialized_engine;
    return 0;
}

int TensorRTBackend::SetInput(void *data, size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t expectedSize = modelLoadOpt_.batch * 3 * modelLoadOpt_.inputHeight * modelLoadOpt_.inputWidth * sizeof(float);
    if(size != expectedSize)
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
    cudaMemcpyAsync(hostBuffer_, deviceBuffers_[1], modelLoadOpt_.batch * modelLoadOpt_.OutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    output_.assign(hostBuffer_, hostBuffer_ + modelLoadOpt_.batch * modelLoadOpt_.OutputSize);
    return output_;
}