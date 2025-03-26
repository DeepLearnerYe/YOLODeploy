#ifndef YOLOV11_DET_HPP
#define YOLOV11_DET_HPP
#include "base_model.hpp"

const int kINPUT_WIDTH = 640;
const int kINPUT_HEIGHT = 640;
struct DetectResult
{
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int classId = 0;
    float confidence = 0;
    std::string className;
    float area = 0;

    std::string toJson() const
    {
        std::ostringstream oss;
        return "{\"classVec\":{ \"point\": [" +
               std::to_string(x0) + "," + std::to_string(y0) + "," +
               std::to_string(x1) + "," + std::to_string(y0) + "," +
               std::to_string(x1) + "," + std::to_string(y1) + "," +
               std::to_string(x0) + "," + std::to_string(y1) + "]" +
               ", \"area\": " + std::to_string(area) +
               ", \"classId\": " + std::to_string(classId) +
               ", \"className\": \"" + className + "\"" +
               ", \"confidence\": " + std::to_string(confidence) +
               ", \"x0\": " + std::to_string(x0) +
               ", \"y0\": " + std::to_string(y0) +
               ", \"x1\": " + std::to_string(x1) +
               ", \"y1\": " + std::to_string(y1) +
               " }}";
    }
};

class YOLOV11Det : public BaseModel<DetectResult>
{
public:
    explicit YOLOV11Det(std::unique_ptr<IInferBackend> backend, float confThreshold = 0.25, float nmsThresold = 0.45);
    ~YOLOV11Det() = default;
    void visualizeRsult(const cv::Mat &img, std::vector<DetectResult> &results);

private:
    std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const cv::Mat &img) override;
    std::vector<DetectResult> PostProcess(const cv::Mat &img, std::vector<float> &modelOutput) override;
    double CalculateFireArea(const cv::Mat &img, const DetectResult det);

    std::vector<std::string> labels_;
    float confThreshold_;
    float nmsThreshold_;
};
#endif