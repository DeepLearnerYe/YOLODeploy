#include <opencv2/opencv.hpp>
#include <fstream>
#include "yolov11_det.hpp"


YOLOV11Det::YOLOV11Det(std::unique_ptr<IInferBackend> backend, const std::string &labelPath, float confThreshold, float nmsThresold)
:BaseModel(std::move(backend)), xFactor_(0), yFactor_(0), confThreshold_(confThreshold), nmsThreshold_(nmsThresold)
{
    std::ifstream file(labelPath);
    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open label file: " + labelPath);
    }
    std::string line;
    while(std::getline(file, line))
    {
        labels_.push_back(line);
    }
}

std::tuple<std::unique_ptr<float[]>, size_t> YOLOV11Det::PreProcess(const Image& img)
{
    cv::Mat image(img.height, img.width, CV_8UC3, img.data);
    int max = std::max(img.height, img.width);
    cv::Mat expandImage = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    cv::Rect roi(0, 0, img.width, img.height);
    image.copyTo(expandImage(roi));
    xFactor_ = image.cols / static_cast<float>(640);
    yFactor_ = image.cols / static_cast<float>(640);

    image_=image.clone();
    cv::imwrite("input.jpg", image);

    // HWC->CHW
    cv::Mat tensor = cv::dnn::blobFromImage(expandImage, 1.0f / 255.f, cv::Size(640, 640), cv::Scalar(), true);

    size_t dataSize = tensor.total();
    std::unique_ptr<float[]> dataPtr(new float[dataSize]);
    std::memcpy(dataPtr.get(), tensor.data, dataSize * sizeof(float));

    return std::make_tuple(std::move(dataPtr), dataSize);
}

std::vector<DetectResult> YOLOV11Det::PostProcess(std::vector<float>& modelOutput)
{
    auto outShape = backend_.get()->GetOutputShape();

    cv::Mat detOutput(outShape.dims[1], outShape.dims[2], CV_32F, const_cast<float*>(backend_.get()->GetOutput().data()));

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for(int i = 0; i < detOutput.cols; ++i)
    {
        cv::Mat classScores = detOutput.col(i).rowRange(4, outShape.dims[1]);
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(classScores, nullptr, &score, nullptr, &classIdPoint);

        if(score > confThreshold_)
        {
            float cx = detOutput.at<float>(0, i);
            float cy = detOutput.at<float>(1, i);
            float ow = detOutput.at<float>(2, i);
            float oh = detOutput.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>(std::max((cx - 0.5 * ow) * xFactor_, 0.0));
            box.y = static_cast<int>(std::max((cy - 0.5 * oh) * yFactor_, 0.0));
            box.width = static_cast<int>(ow * xFactor_);
            box.height = static_cast<int>(oh * yFactor_);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indexes);
    std::cout << "NMS Selected " << indexes.size() << " out of " << boxes.size() << " boxes." << std::endl;


    std::vector<DetectResult> results;
    for (int i = 0; i < indexes.size(); ++i)
    {
        DetectResult result;
        int idx = indexes[i];

        result.x0 = boxes[idx].x;
        result.y0 = boxes[idx].y;
        result.x1 = boxes[idx].x + boxes[idx].width;
        result.y1 = boxes[idx].y + boxes[idx].height;
        result.confidence = confidences[idx];
        result.classId = classIds[idx];
        result.className = labels_[classIds[idx]];
        results.push_back(result);
    }
    
    return results;
}

void YOLOV11Det::visualizeRsult(const Image& img, std::vector<DetectResult>& results)
{
    cv::Mat image(img.height, img.width, CV_8UC3, img.data);
    for(auto &elem: results)
    {
        cv::Rect box;
        box.x = elem.x0;
        box.y = elem.y0;
        box.width = elem.x1 - elem.x0;
        box.height = elem.y1 - elem.y0;
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2, 8);
    }
    cv::imwrite("output.jpg", image);
}