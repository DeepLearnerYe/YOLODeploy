#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <vector>
#include <mutex>
#include "types.hpp"

class IInferBackend
{
public:
    IInferBackend(const ModelLoadOpt& modelLoadOpt);
    virtual ~IInferBackend() = default;
    virtual int Initialize() = 0;
    virtual int SetInput(void* data, size_t size) = 0;
    virtual int Infer() = 0;
    virtual const std::vector<float>& GetOutput() = 0;
protected:
    ModelLoadOpt modelLoadOpt_;
    std::mutex mutex_;
    float* hostBuffer_;
    void* deviceBuffers_[2];
    std::vector<float> output_;
};
#endif