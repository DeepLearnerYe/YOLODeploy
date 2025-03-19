#ifndef YOLOV11_DET_HPP
#define YOLOV11_DET_HPP
#include "base_model.hpp"

struct DetectResult
{
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int classId = 0;
    float confidence = 0;
    std::string className;
};

class YOLOV11Det:public BaseModel<DetectResult>
{
public:
    explicit YOLOV11Det(std::unique_ptr<IInferBackend> backend, const std::string &lablePath, float confThreshold = 0.25, float nmsThresold = 0.45);
    ~YOLOV11Det() = default;
    void visualizeRsult(const Image& img, std::vector<DetectResult>& results);

private:
    std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const Image& img) override;
    std::vector<DetectResult> PostProcess(std::vector<float>& modelOutput) override;

    std::vector<std::string> labels_;
    float xFactor_;
    float yFactor_;
    float confThreshold_;
    float nmsThreshold_;
    // 调试使用
    cv::Mat image_;
};
#endif