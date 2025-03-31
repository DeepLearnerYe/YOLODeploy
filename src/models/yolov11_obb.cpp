#include <fstream>
#include <algorithm>
#include "yolov11_obb.hpp"
#include <numeric>
#define pi acos(-1)

YOLOV11Obb::YOLOV11Obb(std::unique_ptr<IInferBackend> backend, float confThreshold, float nmsThresold)
    : BaseModel(std::move(backend)), confThreshold_(confThreshold), nmsThreshold_(nmsThresold)
{
    labels_ = backend_.get()->GetLabels();
}

std::tuple<std::unique_ptr<float[]>, size_t> YOLOV11Obb::PreProcess(const cv::Mat &img)
{
    int max = std::max(img.cols, img.rows);
    cv::Mat expandImage = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    expandImage.setTo(cv::Scalar(114, 114, 114));
    cv::Rect roi(0, 0, img.cols, img.rows);
    img.copyTo(expandImage(roi));

    // HWC->CHW
    cv::Mat tensor = cv::dnn::blobFromImage(expandImage, 1.0f / 255.f, cv::Size(kINPUT_WIDTH, kINPUT_HEIGHT), cv::Scalar(), true);

    size_t dataSize = tensor.total();
    std::unique_ptr<float[]> dataPtr(new float[dataSize]);
    std::memcpy(dataPtr.get(), tensor.data, dataSize * sizeof(float));

    return std::make_tuple(std::move(dataPtr), dataSize);
}

std::vector<ObbResult> YOLOV11Obb::PostProcess(const cv::Mat &img, std::vector<float> &modelOutput)
{
    auto outShape = backend_.get()->GetOutputShape();
    float imageSize = static_cast<float>(std::max(img.cols, img.rows));
    float xFactor = imageSize / kINPUT_WIDTH;
    float yFactor = imageSize / kINPUT_HEIGHT;

    cv::Mat detOutput(outShape.dims[1], outShape.dims[2], CV_32F, const_cast<float *>(backend_.get()->GetOutput().data()));

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    for (int i = 0; i < detOutput.cols; ++i)
    {
        cv::Mat classScores = detOutput.col(i).rowRange(4, outShape.dims[1] - 1);
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(classScores, nullptr, &score, nullptr, &classIdPoint);

        if (score > confThreshold_)
        {
            float cx = detOutput.at<float>(0, i);
            float cy = detOutput.at<float>(1, i);
            float ow = detOutput.at<float>(2, i);
            float oh = detOutput.at<float>(3, i);
            float angle = detOutput.at<float>(outShape.dims[1] - 1, i) * 180 / pi;
            if (angle < 0)
            {
                angle += 360;
            }
            else if (angle > 360)
            {
                angle -= 360;
            }
            cv::RotatedRect box(cv::Point2f(cx * xFactor, cy * yFactor), cv::Size2f(ow * xFactor, oh * yFactor), angle);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indexes);

    // transform to the specific format
    std::vector<ObbResult> results;
    for (int idx : indexes)
    {
        cv::Point2f vertices[4];
        boxes[idx].points(vertices);
        RotatedRect rotatedRec;
        rotatedRec.leftTopx = vertices[0].x;
        rotatedRec.leftTopy = vertices[0].y;
        rotatedRec.rightTopx = vertices[1].x;
        rotatedRec.rightTopy = vertices[1].y;
        rotatedRec.rightBottomx = vertices[2].x;
        rotatedRec.rightBottomy = vertices[2].y;
        rotatedRec.leftBottomx = vertices[3].x;
        rotatedRec.leftBottomy = vertices[3].y;

        ObbResult result;
        result.rotatedRec = rotatedRec;
        result.classId = classIds[idx];
        result.className = labels_[classIds[idx]];
        result.confidence = confidences[idx];
        results.push_back(result);
    }

    return results;
}

void YOLOV11Obb::visualizeRsult(const cv::Mat &image, std::vector<ObbResult> &results)
{
    for (auto &elem : results)
    {
        cv::line(image, cv::Point2f(elem.rotatedRec.leftTopx, elem.rotatedRec.leftTopy),
                 cv::Point2f(elem.rotatedRec.rightTopx, elem.rotatedRec.rightTopy),
                 cv::Scalar(0, 255, 0), 2);

        cv::line(image, cv::Point2f(elem.rotatedRec.rightTopx, elem.rotatedRec.rightTopy),
                 cv::Point2f(elem.rotatedRec.rightBottomx, elem.rotatedRec.rightBottomy),
                 cv::Scalar(0, 255, 0), 2);

        cv::line(image, cv::Point2f(elem.rotatedRec.rightBottomx, elem.rotatedRec.rightBottomy),
                 cv::Point2f(elem.rotatedRec.leftBottomx, elem.rotatedRec.leftBottomy),
                 cv::Scalar(0, 255, 0), 2);

        cv::line(image, cv::Point2f(elem.rotatedRec.leftBottomx, elem.rotatedRec.leftBottomy),
                 cv::Point2f(elem.rotatedRec.leftTopx, elem.rotatedRec.leftTopy),
                 cv::Scalar(0, 255, 0), 2);

        std::string text = std::to_string(elem.confidence) + ": " + elem.className;
        cv::putText(image, text, cv::Point(elem.rotatedRec.rightTopx, elem.rotatedRec.rightTopy), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("rotatedOut.jpg", image);
}