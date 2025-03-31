#include <fstream>
#include <algorithm>
#include "yolov11_obb.hpp"

#include <sys/stat.h>
#include <sys/types.h>
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

std::vector<std::vector<float>> batchProbiou(const std::vector<RotatedRect>& obb1, const std::vector<RotatedRect>& obb2, float eps = 1e-6f) {
    size_t N = obb1.size();
    size_t M = obb2.size();
    std::vector<std::vector<float>> iou_matrix(N, std::vector<float>(M, 0.0f));
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float iou = 1.0f; // Placeholder: Implement actual IoU computation
            iou_matrix[i][j] = iou;
        }
    }
    return iou_matrix;
}

std::vector<int> nmsRotated(const std::vector<RotatedRect>& boxes, const std::vector<float>& scores, float iou_thres) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<RotatedRect> sorted_boxes;
    for (int idx : indices) {
        sorted_boxes.push_back(boxes[idx]);
    }
    
    std::vector<std::vector<float>> ious = batchProbiou(sorted_boxes, sorted_boxes);
    std::vector<int> keep;
    int n = sorted_boxes.size();
    
    for (int j = 0; j < n; ++j) {
        bool keep_j = true;
        for (int i = 0; i < j; ++i) {
            if (ious[i][j] >= iou_thres) {
                keep_j = false;
                break;
            }
        }
        if (keep_j) keep.push_back(j);
    }
    
    std::vector<int> keep_indices;
    for (int k : keep) {
        keep_indices.push_back(indices[k]);
    }
    return keep_indices;
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
    std::cout << "detOutput.rows -1 = " << detOutput.rows -1 << std::endl;
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
            float angle = detOutput.at<float>(6, i);
            // if (angle>=0.5*pi && angle <= 0.75*pi)
            // {
            //     angle=angle-pi;
            // }
            cv::RotatedRect box(cv::Point2f(cx * xFactor, cy * yFactor), cv::Size2f(ow * xFactor, oh * yFactor), angle*180/pi);
            // cv::RotatedRect box(cv::Point2f(cx, cy), cv::Size2f(ow, oh), theta);
            // RotatedRect box;
            // box.x = static_cast<double>(cx * xFactor);
            // box.y = static_cast<double>(cy * yFactor);
            // box.w = static_cast<double>(ow * xFactor);
            // box.h = static_cast<double>(oh * yFactor);
            // box.angle = angle*180.0/pi;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.y);
            confidences.push_back(score);
        }
    }

    
   
    // auto indexes = nmsRotated(boxes, confidences, 0.4);
    std::vector<int> indexes;
    //  indexes.resize(19);
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indexes);

    std::cout << "selected: " << indexes.size() << " of " << boxes.size() << std::endl;

    std::vector<cv::RotatedRect> cvBoxes;
    for(auto &rec: boxes)
    {
        cvBoxes.push_back(rec);
        // cvBoxes.emplace_back(cv::Point2f(rec.x, rec.y), cv::Size2f(rec.w, rec.h), rec.angle);

    }
    for (int idx : indexes)
    {
        cv::Point2f vertices[4];
        cvBoxes[idx].points(vertices);
        for (int i = 0; i < 4; ++i)
        {
            std::cout << "Vertex " << i << ": " << vertices[i] << std::endl;
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        // std::cout << "x = " << boxes[idx].x << std::endl;
        // std::cout << "y = " << boxes[idx].y << std::endl;
        // std::cout << "w = " << boxes[idx].w << std::endl;
        // std::cout << "h = " << boxes[idx].h << std::endl;
        // std::cout << "theta = " << boxes[idx].angle << std::endl;
    }
    cv::imwrite("rotatedOut.jpg", img);

    std::vector<ObbResult> results;
    // for (int i = 0; i < indexes.size(); ++i)
    // {
    //     ObbResult result;
    //     int idx = indexes[i];

    //     result.x0 = boxes[idx].x;
    //     result.y0 = boxes[idx].y;
    //     result.x1 = boxes[idx].x + boxes[idx].width;
    //     result.y1 = boxes[idx].y + boxes[idx].height;
    //     result.confidence = confidences[idx];
    //     result.classId = classIds[idx];
    //     result.className = labels_[classIds[idx]];

    //     results.push_back(result);
    // }

    return results;
}

// void YOLOV11Obb::visualizeRsult(const cv::Mat &image, std::vector<ObbResult> &results)
// {
//     for (auto &elem : results)
//     {
//         cv::Rect box;
//         box.x = elem.x0;
//         box.y = elem.y0;
//         box.width = elem.x1 - elem.x0;
//         box.height = elem.y1 - elem.y0;
//         cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2, 8);
//         std::string text = std::to_string(elem.classId) + ": " + elem.className;
//         cv::putText(image, text, cv::Point(elem.x0, elem.y0 - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
//     }
//     cv::imwrite("detOutput.jpg", image);
// }