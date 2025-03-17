#include <string>

enum ModelType
{
    ENGINE,
    OM
};

struct ModelLoadOpt
{
    std::string modelPath;
    ModelType modelType;
    unsigned short deviceId = 0;
    size_t batch = 1;
    size_t inputWidth = 0;
    size_t inputHeight = 0;
    size_t OutputSize = 0;
};

struct Image
{
    void *data;
    int width = 0;
    int height = 0;
};