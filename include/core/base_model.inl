
#include "base_model.hpp"

template <typename ResultType>
BaseModel<ResultType>::BaseModel(std::unique_ptr<IInferBackend> backend)
    : backend_(std::move(backend)) 
{
        backend_->Initialize();
}

template <typename ResultType>
std::vector<ResultType> BaseModel<ResultType>::Predict(const cv::Mat &image)
{
    // std::lock_guard<std::mutex> lock(mutex_);
    TimerUtils::Start("preprocess");
    auto [input_data, input_size] = PreProcess(image);
    TimerUtils::Stop("preprocess");

    TimerUtils::Start("infer");
    // TimerUtils::Start("SetInput");
    backend_->SetInput(input_data.get(), input_size);
    // TimerUtils::Stop("SetInput");
    backend_->Infer();
    // TimerUtils::Start("GetOutput");
    auto output = backend_->GetOutput();
    // TimerUtils::Stop("GetOutput");
    TimerUtils::Stop("infer");

    TimerUtils::Start("postprocess");
    auto result = PostProcess(image, output);
    TimerUtils::Stop("postprocess");
    return result;
}