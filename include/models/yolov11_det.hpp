#include "base_model.hpp"

struct DetectResult
{
    
};

class YOLOV11Det:public BaseModel<DetectResult>
{
public:
    explicit YOLOV11Det(std::unique_ptr<IInferBackend> backend);
    ~YOLOV11Det() = default;

protected:
    std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const Image& img) override;
    DetectResult PostProcess(void* output, size_t size) override;
};