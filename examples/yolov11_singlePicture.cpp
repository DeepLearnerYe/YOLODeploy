#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov11_det.hpp"
#include "tensorrt_backend.hpp"


void singlePicture(std::string imagePath)
{
    ModelLoadOpt opt;
    opt.modelPath = "/root/host_map/yolov11/ryxw.engine";
    opt.modelType = ModelType::ENGINE;
    std::string label = "/root/host_map/yolov11/ryxw.names";
    
    std::unique_ptr<TensorRTBackend> backend(new TensorRTBackend(opt));
    YOLOV11Det det(std::move(backend), label);

    cv::Mat mat = cv::imread(imagePath);
    Image image;
    image.data = mat.data;
    image.width = mat.cols;
    image.height = mat.rows;
    auto result = det.Predict(image);
    det.visualizeRsult(image, result);
}

int main(int argc, char**argv)
{
    if(argc != 2)
    {
        std::cout << "nums of params incorrect " << std::endl;
        return 0;
    }
    singlePicture(argv[1]);

    return 0;
}