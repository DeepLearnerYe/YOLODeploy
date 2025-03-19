#ifndef TYPES_HPP
#define TYPES_HPP
#include <string>
#include <vector>

enum ModelType
{
    ENGINE,
    OM
};

struct ModelLoadOpt
{
    std::string modelPath;
    std::string labelPath;
    ModelType modelType;
    unsigned short deviceId = 0;
    size_t batch = 1;
    size_t inputHeight = 0;
    size_t inputWidth = 0;
    size_t OutputSize = 0;
};

struct Image
{
    void *data;
    int width = 0;
    int height = 0;
};

struct Shape
{
    std::vector<int32_t> dims;
};
#endif