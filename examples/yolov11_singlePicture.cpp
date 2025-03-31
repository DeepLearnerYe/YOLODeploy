#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov11_det.hpp"
#include "yolov11_cls.hpp"
#include "yolov11_obb.hpp"
#include "tensorrt_backend.hpp"
#include "utils.hpp"

void singlePicture(std::string imagePath)
{
    ModelLoadOpt opt;
    opt.modelPath = "/root/host_map/yolov11/lib_nvidia_ryxw.so.1.1.20250319";
    opt.modelType = ModelType::ENGINE;
    
    std::unique_ptr<TensorRTBackend> backend(new TensorRTBackend(opt));
    YOLOV11Det det(std::move(backend));

    std::string base64Image;
    std::vector<uchar> imageContent;
    auto ret = ReadFile(imagePath, base64Image);
    if(ret < 0)
    {
        return;
    }
    Base64Decode(base64Image, imageContent);

    cv::Mat mat = cv::imdecode(imageContent, cv::IMREAD_COLOR);
    auto result = det.Predict(mat);
    det.visualizeRsult(mat, result);
}

void singlePicture2(std::string imagePath)
{
    ModelLoadOpt opt;
    opt.modelPath = "/root/host_map/yolov11/lib_nvidia_person_cls.so.1.1.20250326";
    opt.modelType = ModelType::ENGINE;
    
    std::unique_ptr<TensorRTBackend> backend(new TensorRTBackend(opt));
    YOLOV11Cls det(std::move(backend));

    std::string base64Image;
    std::vector<uchar> imageContent;
    auto ret = ReadFile(imagePath, base64Image);
    if(ret < 0)
    {
        return;
    }
    Base64Decode(base64Image, imageContent);

    cv::Mat mat = cv::imdecode(imageContent, cv::IMREAD_COLOR);
    
    auto result = det.Predict(mat);
    det.visualizeRsult(mat, result);
}

void singlePicture3(std::string imagePath)
{
    ModelLoadOpt opt;
    opt.modelPath = "/root/host_map/yolov11/lib_nvidia_jjd.so.1.1.20250328";
    opt.modelType = ModelType::ENGINE;
    
    std::unique_ptr<TensorRTBackend> backend(new TensorRTBackend(opt));
    YOLOV11Obb det(std::move(backend));

    std::string base64Image;
    std::vector<uchar> imageContent;
    auto ret = ReadFile(imagePath, base64Image);
    if(ret < 0)
    {
        return;
    }
    Base64Decode(base64Image, imageContent);

    cv::Mat mat = cv::imdecode(imageContent, cv::IMREAD_COLOR);
    auto result = det.Predict(mat);
    det.visualizeRsult(mat, result);

//     for(auto &elem:result)
//     {
//         std::cout << elem.toJson() << std::endl;
//     }
}

int main(int argc, char**argv)
{
    if(argc != 2)
    {
        std::cout << "nums of params incorrect " << std::endl;
        return 0;
    }
    singlePicture3(argv[1]);

    return 0;
}