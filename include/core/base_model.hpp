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
    std::vector<ResultType> Predict(const Image& image);

protected:
    virtual std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const Image& img) = 0;

    virtual std::vector<ResultType> PostProcess(const Image& img, std::vector<float>& modelOutput) = 0;

    std::unique_ptr<IInferBackend> backend_;
    std::mutex mutex_;
};

#include "base_model.inl"
#endif