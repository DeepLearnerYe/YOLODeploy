#include "backend.hpp"

IInferBackend::IInferBackend(const ModelLoadOpt& modelLoadOpt)
:modelLoadOpt_(modelLoadOpt), hostBuffer_(nullptr), deviceBuffers_{nullptr, nullptr}
{

}