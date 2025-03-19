#include "backend.hpp"

IInferBackend::IInferBackend(const ModelLoadOpt& modelLoadOpt)
:modelLoadOpt_(modelLoadOpt), hostBuffer_(nullptr), deviceBuffers_{nullptr, nullptr}
{

}

std::vector<std::string> &IInferBackend::GetLabels()
{
    return labels_;
}