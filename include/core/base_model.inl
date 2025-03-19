
#include "base_model.hpp"

template <typename ResultType>
BaseModel<ResultType>::BaseModel(std::unique_ptr<IInferBackend> backend)
    : backend_(std::move(backend)) 
{
        backend_->Initialize();
}

template <typename ResultType>
std::vector<ResultType> BaseModel<ResultType>::Predict(const Image &image)
{
    auto [input_data, input_size] = PreProcess(image);

    backend_->SetInput(input_data.get(), input_size);

    backend_->Infer();

    auto output = backend_->GetOutput();

    return PostProcess(output);
}