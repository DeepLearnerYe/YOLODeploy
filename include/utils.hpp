#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <dlfcn.h>
#include <string>
#include <vector>

typedef unsigned char *(*GetModelDataFunc)(const char *);
typedef unsigned int (*GetModelLenFunc)();
typedef const char **(*GetClassName)();

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
#endif