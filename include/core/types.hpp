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
    int input_width = 0;
    int input_height = 0;
};

struct Image
{
    void *data;
    int width = 0;
    int height = 0;
};