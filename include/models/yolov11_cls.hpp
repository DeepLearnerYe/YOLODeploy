#ifndef YOLOV11_CLS_HPP
#define YOLOV11_CLS_HPP
#include "base_model.hpp"


struct CLSResult
{
    int classId = 0;
    float confidence = 0;
    std::string className;
};

class YOLOV11Cls : public BaseModel<CLSResult>
{
public:
    explicit YOLOV11Cls(std::unique_ptr<IInferBackend> backend, float confThreshold = 0.25, float nmsThresold = 0.45);
    ~YOLOV11Cls() = default;
    void visualizeRsult(const cv::Mat &img, std::vector<CLSResult> &results);

private:
    std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const cv::Mat &img) override;
    std::vector<CLSResult> PostProcess(const cv::Mat& img, std::vector<float> &modelOutput) override;
    std::vector<float> Softmax(std::vector<float> &modelOutput);
    std::vector<int> TopK(const std::vector<float>& vec, int k);

    std::vector<std::string> labels_;
    float confThreshold_;
    float nmsThreshold_;
    static const int kINPUT_WIDTH = 224;
    static const int kINPUT_HEIGHT = 224;
};
#endif