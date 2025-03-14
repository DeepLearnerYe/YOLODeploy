#include "types.hpp"
class IInferBackend
{
public:
    IInferBackend() = default;
    virtual ~IInferBackend() = default;
    virtual void LoadModel(ModelLoadOpt &modelLoadOpt) = 0;
    virtual void SetInput(void *data, size_t size) = 0;
    virtual void Infer() = 0;
    virtual void *GetOutput() = 0;
    virtual size_t GetOutputSize() = 0;
};