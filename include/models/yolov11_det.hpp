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
    float area = 0;

    std::string toJson() const
    {
        std::ostringstream oss;
        oss << "{"
            << "\"classVec\":{ \"point\": ["
            << x0 << "," << y0 << "," << x1 << "," << y0 << ","
            << x1 << "," << y1 << "," << x0 << "," << y1 << "],"
            << "\"area\": " << area << ","
            << "\"classId\": " << classId << ","
            << "\"className\": \"" << className << "\","
            << "\"confidence\": " << confidence << ","
            << "\"x0\": " << x0 << ","
            << "\"y0\": " << y0 << ","
            << "\"x1\": " << x1 << ","
            << "\"y1\": " << y1
            << " }}";

        return oss.str();
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
    static const int kINPUT_WIDTH = 640;
    static const int kINPUT_HEIGHT = 640;
};
#endif