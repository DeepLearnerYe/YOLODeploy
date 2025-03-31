#ifndef YOLOV11_OBB_HPP
#define YOLOV11_OBB_HPP
#include "base_model.hpp"

struct RotatedRect
{
    float leftTopx = 0;
    float leftTopy = 0;
    float rightTopx = 0;
    float rightTopy = 0;
    float rightBottomx = 0;
    float rightBottomy = 0;
    float leftBottomx = 0;
    float leftBottomy = 0;
};

struct ObbResult
{
    RotatedRect rotatedRec;
    int classId = 0;
    float confidence = 0;
    std::string className;

    std::string toJson() const
    {
        std::ostringstream oss;
        oss << "{"
            << "\"classVec\":{ \"point\": ["
            << rotatedRec.leftTopx << "," << rotatedRec.leftTopy << "," << rotatedRec.rightTopx << "," << rotatedRec.rightTopy << ","
            << rotatedRec.rightBottomx << "," << rotatedRec.rightBottomy << "," << rotatedRec.leftBottomx << "," << rotatedRec.leftBottomy << "],"
            << "\"classId\": " << classId << ","
            << "\"className\": \"" << className << "\","
            << "\"confidence\": " << confidence
            << " }}";

        return oss.str();
    }
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

    std::vector<std::string> labels_;
    float confThreshold_;
    float nmsThreshold_;
    static const int kINPUT_WIDTH = 640;
    static const int kINPUT_HEIGHT = 640;
};
#endif