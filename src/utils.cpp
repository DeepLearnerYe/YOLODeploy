#include "utils.hpp"

int OpenLibrary(const std::string &libPath, unsigned char *&modelPtr, unsigned int &modelLen, std::vector<std::string> &labels)
{
	// 打开动态库
	void *modelHandle_ = dlopen(libPath.c_str(), RTLD_LAZY);
	if (!modelHandle_)
	{
		std::cerr << "dlopen failed: " << dlerror() << std::endl;
		return -1;
	}

	// 获取动态库里的函数
	GetModelDataFunc getModelData = (GetModelDataFunc)dlsym(modelHandle_, "getModelData");
	GetModelLenFunc getModelLen = (GetModelLenFunc)dlsym(modelHandle_, "getModelLen");
	GetClassName getClassName = (GetClassName)dlsym(modelHandle_, "getClassName");
	GetModelLenFunc getClassNum = (GetModelLenFunc)dlsym(modelHandle_, "getClassNum");

	// 错误处理
	const char *dlsymError = dlerror();
	if (dlsymError)
	{
		std::cerr << "Cannot load symbol 'processModel': " << dlsymError << '\n';
		dlclose(modelHandle_);
		return -2;
	}

	// 获取模型数据
	const char *key = "jiayang2024";
	modelPtr = getModelData(key);
	modelLen = getModelLen();
	// 获取模型标签
	for (int i = 0; i < getClassNum(); ++i)
	{
		labels.push_back(std::string(getClassName()[i]));
	}

	return 0;
}

int ReadFile(const std::string filePath, std::string &content)
{
    std::fstream file(filePath, std::ios::in);
    if(!file.is_open())
    {
        std::cout << "ReadFile from "<< filePath << " failed " << std::endl;
        return -1;
    }

    content = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return 0;
}

bool isBase64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

int Base64Decode(const std::string &encodedString, std::vector<unsigned char> &deco)
{
    size_t inLen = encodedString.size();
    size_t i = 0;
    size_t j = 0;
    size_t in = 0;
    unsigned char charArray4[4], charArray3[3];

    while (inLen-- && (encodedString[in] != '=') && isBase64(encodedString[in]))
    {
        // access 4 bytes
        charArray4[i++] = encodedString[in];
        in++;
        if (i == 4)
        {
            for (i = 0; i < 4; ++i)
            {
                charArray4[i] = KBASE64.find(charArray4[i]);
            }

            charArray3[0] = (charArray4[0] << 2) | ((charArray4[1] & 0b00110000) >> 4);
            charArray3[1] = ((charArray4[1] & 0b00001111) << 4) | ((charArray4[2] & 0b00111100) >> 2);
            charArray3[2] = ((charArray4[2] & 0b00000011) << 6) | charArray4[3];

            for (i = 0; (i < 3); i++)
                deco.push_back(charArray3[i]);
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j < 4; j++)
            charArray4[j] = 0;

        for (j = 0; j < 4; j++)
            charArray4[j] = KBASE64.find(charArray4[j]);

        charArray3[0] = (charArray4[0] << 2) | ((charArray4[1] & 0b00110000) >> 4);
        charArray3[1] = ((charArray4[1] & 0b00001111) << 4) | ((charArray4[2] & 0b00111100) >> 2);
        charArray3[2] = ((charArray4[2] & 0b00000011) << 6) | charArray4[3];

        for (j = 0; (j < i - 1); j++)
            deco.push_back(charArray3[j]);
    }

    return 0;
}