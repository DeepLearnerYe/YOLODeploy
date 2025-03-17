#ifndef BASE_MODEL_HPP
#define BASE_MODEL_HPP

#include <memory>
#include <tuple>
#include "backend.hpp"
#include "types.hpp"

template<typename ResultType>
class BaseModel
{
public:
    explicit BaseModel(std::unique_ptr<IInferBackend> backend);
    ~BaseModel() = default;
    ResultType Predict(const Image& image);

protected:
    virtual std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const Image& img) = 0;

    virtual ResultType PostProcess(void* output, size_t size) = 0;

    std::unique_ptr<IInferBackend> backend_;
};

#include "base_model.inl"
#endif