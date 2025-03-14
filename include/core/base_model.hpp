
template<typename ResultType>
class BaseModel
{
public:
    explicit BaseModel(std::unique_ptr<IInferBackend> backend) : backend_(std::move(backend)) {}
    
    ResultType Predict(const Image& image) {
        // 1. 预处理生成输入数据
        auto [input_data, input_size] = PreProcess(image);
        
        // 2. 设置输入到后端
        backend_->SetInput(input_data.get(), input_size);
        
        // 3. 执行推理
        try{
            backend_->Infer();
        }catch(...){

        }
        
        // 4. 获取并解析输出
        void* output = backend_->GetOutput();
        size_t output_size = backend_->GetOutputSize();
        return PostProcess(output, output_size);
    }

protected:
    virtual std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const Image& img) {
        // YOLO特有的预处理逻辑
        // 根据模型输入尺寸计算
        size_t input_size = model_input_width_ * model_input_height_ * 3 * sizeof(float);
        auto processed = std::make_unique<float[]>(input_size);
        return {std::move(processed), 1024};
    }

    virtual ResultType PostProcess(void* output, size_t size) {
        // YOLO特有的后处理逻辑
        return {};
    }

    std::unique_ptr<IInferBackend> backend_;
};