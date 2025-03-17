#include <opencv2/opencv.hpp>
#include "yolov11_det.hpp"

std::tuple<std::unique_ptr<float[]>, size_t> YOLOV11Det::PreProcess(const Image& img)
{
    cv::Mat image(img.height, img.width, CV_8UC3, img.data);
    int max = std::max(img.height, img.width);
    cv::Mat expandImage = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    cv::Rect roi(0, 0, img.width, img.height);
    image.copyTo(expandImage(roi));

    cv::Mat tensor = cv::dnn::blobFromImage(expandImage, 1.0f / 255.f, cv::Size(640, 640), cv::Scalar(), true);

    size_t dataSize = tensor.total();
    std::unique_ptr<float[]> dataPtr(new float[dataSize]);
    std::memcpy(dataPtr.get(), tensor.data, dataSize * sizeof(float));

    return std::make_tuple(std::move(dataPtr), dataSize);
}

DetectResult YOLOV11Det::PostProcess(void* output, size_t size)
{
    DetectResult result;
    return result;
}