#ifndef YOLOV11_OBB_HPP
#define YOLOV11_OBB_HPP
#include "base_model.hpp"

struct RotatedRect
{
    float x = 0;
    float y = 0;
    float w = 0;
    float h = 0;
    float angle = 0;
};

struct ObbResult
{
    RotatedRect rotatedRec;
    int classId = 0;
    float confidence = 0;
    std::string className;
};

class YOLOV11Obb : public BaseModel<ObbResult>
{
public:
    explicit YOLOV11Obb(std::unique_ptr<IInferBackend> backend, float confThreshold = 0.25, float nmsThresold = 0.45);
    ~YOLOV11Obb() = default;
    void visualizeRsult(const cv::Mat &img, std::vector<ObbResult> &results);

private:
    std::tuple<std::unique_ptr<float[]>, size_t> PreProcess(const cv::Mat &img) override;
    std::vector<ObbResult> PostProcess(const cv::Mat &img, std::vector<float> &modelOutput) override;
    double CalculateFireArea(const cv::Mat &img, const ObbResult det);

    std::vector<std::string> labels_;
    float confThreshold_;
    float nmsThreshold_;
    static const int kINPUT_WIDTH = 640;
    static const int kINPUT_HEIGHT = 640;
};
#endif