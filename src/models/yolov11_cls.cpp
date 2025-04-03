#include <fstream>
#include <algorithm>
#include <numeric>
#include "yolov11_cls.hpp"


YOLOV11Cls::YOLOV11Cls(std::unique_ptr<IInferBackend> backend, float confThreshold, float nmsThresold)
    : BaseModel(std::move(backend)), confThreshold_(confThreshold), nmsThreshold_(nmsThresold)
{
    labels_ = backend_.get()->GetLabels();
}

std::tuple<std::unique_ptr<float[]>, size_t> YOLOV11Cls::PreProcess(const cv::Mat &img)
{
    // center crop
    int height = img.rows;
    int width = img.cols;
    int min = std::min(height, width);
    int top = (height - min) / 2;
    int left = (width - min) / 2;
    cv::Mat image = img(cv::Rect(left, top, min, min));
    cv::resize(image, image, cv::Size(kINPUT_HEIGHT, kINPUT_WIDTH), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    float *output = new float[3 * kINPUT_HEIGHT * kINPUT_WIDTH];

    // HWC -> CHW
    for (int c = 0; c < 3; ++c)
    {
        for (int row = 0; row < kINPUT_HEIGHT; ++row)
        {
            for (int col = 0; col < kINPUT_WIDTH; ++col)
            {
                output[col + row * kINPUT_WIDTH + c * kINPUT_HEIGHT * kINPUT_WIDTH] =
                    channels[c].at<float>(row, col);
            }
        }
    }
    std::unique_ptr<float[]> dataPtr(output);
   
    return std::make_tuple(std::move(dataPtr), 3 * kINPUT_HEIGHT * kINPUT_WIDTH);
}

std::vector<float> YOLOV11Cls::Softmax(std::vector<float> &modelOutput)
{
    std::vector<float> res;
    float t = 0, sum = 0;
    // avoid value overflow
    float max = *std::max_element(modelOutput.begin(), modelOutput.end());
    for (int i = 0; i < modelOutput.size(); ++i)
    {
        t = expf(modelOutput[i] - max);
        res.push_back(t);
        sum += t;
    }
    for (int i = 0; i < modelOutput.size(); ++i)
    {
        res[i] /= sum;
    }
    return res;
}

std::vector<int> YOLOV11Cls::TopK(const std::vector<float> &vec, int k)
{
    std::vector<int> topkIndex;
    std::vector<size_t> vecIndex(vec.size());
    std::iota(vecIndex.begin(), vecIndex.end(), 0);
    std::sort(vecIndex.begin(), vecIndex.end(),
              [&vec](size_t index1, size_t index2)
              { return vec[index1] > vec[index2]; });
    int kNum = std::min<int>(vec.size(), k);
    for (int i = 0; i < kNum; ++i)
    {
        topkIndex.push_back(vecIndex[i]);
    }
    return topkIndex;
}

std::vector<CLSResult> YOLOV11Cls::PostProcess(const cv::Mat &img, std::vector<float> &modelOutput)
{
    auto res = Softmax(modelOutput);
    auto topk_idx = TopK(res, 1);

    // transform to the specific format
    CLSResult result;
    result.classId = topk_idx[0];
    result.confidence = res[topk_idx[0]];
    result.className = labels_[topk_idx[0]];

    return {result};
}
int count = 0;
void YOLOV11Cls::visualizeRsult(const cv::Mat &image, std::vector<CLSResult> &results)
{
    for (auto &elem : results)
    {
        std::string text = std::to_string(elem.confidence) + ": " + elem.className;
        cv::putText(image, text, cv::Point(30, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    std::string name = "clsOutput" + std::to_string(count++) + ".jpg";
    cv::imwrite(name, image);
}