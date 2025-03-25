#include <opencv2/opencv.hpp>
#include <fstream>
#include "yolov11_det.hpp"

#include <sys/stat.h>
#include <sys/types.h>


YOLOV11Det::YOLOV11Det(std::unique_ptr<IInferBackend> backend, float confThreshold, float nmsThresold)
:BaseModel(std::move(backend)), confThreshold_(confThreshold), nmsThreshold_(nmsThresold)
{
    labels_ = backend_.get()->GetLabels();
}

std::tuple<std::unique_ptr<float[]>, size_t> YOLOV11Det::PreProcess(const Image& img)
{
    cv::Mat image(img.height, img.width, CV_8UC3, img.data);
    // std::cout << "img.width= " << img.width << "img.height= " << img.height << std::endl;
    // std::cout << "img.cols= " << image.cols << "img.rows= " << image.rows << std::endl;
    int max = std::max(img.height, img.width);
    cv::Mat expandImage = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    expandImage.setTo(cv::Scalar(114, 114, 114));
    cv::Rect roi(0, 0, img.width, img.height);
    image.copyTo(expandImage(roi));
    // cv::imwrite("resize.jpg", expandImage);

    // image_=image.clone();
    // std::string folder = "testImg";
    // struct stat info;
    // if(stat(folder.c_str(), &info) != 0)
    // {
    //     mkdir(folder.c_str(), 0777);
    // }
    // std::string inputName = folder + "/input" + std::to_string(count_++) + ".jpg";
    // cv::imwrite(inputName, image);

    // HWC->CHW
    cv::Mat tensor = cv::dnn::blobFromImage(expandImage, 1.0f / 255.f, cv::Size(kINPUT_WIDTH, kINPUT_HEIGHT), cv::Scalar(), true);

    size_t dataSize = tensor.total();
    std::unique_ptr<float[]> dataPtr(new float[dataSize]);
    std::memcpy(dataPtr.get(), tensor.data, dataSize * sizeof(float));

    return std::make_tuple(std::move(dataPtr), dataSize);
}

std::vector<DetectResult> YOLOV11Det::PostProcess(const Image &img, std::vector<float>& modelOutput)
{
    auto outShape = backend_.get()->GetOutputShape();
    float imageSize = static_cast<float>(std::max(img.width, img.height));
    float xFactor = imageSize / kINPUT_WIDTH;
    float yFactor = imageSize / kINPUT_HEIGHT;

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
            box.x = static_cast<int>(std::max((cx - 0.5 * ow) * xFactor, 0.0));
            box.y = static_cast<int>(std::max((cy - 0.5 * oh) * yFactor, 0.0));
            box.width = static_cast<int>(ow * xFactor);
            box.height = static_cast<int>(oh * yFactor);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indexes);
    // std::cout << "NMS Selected " << indexes.size() << " out of " << boxes.size() << " boxes." << std::endl;

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
        std::string text = std::to_string(elem.classId) + ": " + elem.className;
        cv::putText(image, text, cv::Point(elem.x0, elem.y0 - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    // std::string outputName = "testImg/output" + std::to_string(count_) + ".jpg";
    // cv::imwrite(outputName, image);
}