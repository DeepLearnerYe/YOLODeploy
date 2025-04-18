#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <chrono>

typedef unsigned char *(*GetModelDataFunc)(const char *);
typedef unsigned int (*GetModelLenFunc)();
typedef const char **(*GetClassName)();

int OpenLibrary(const std::string &libPath, unsigned char *&modelPtr, unsigned int &modelLen, std::vector<std::string> &labels);

int ReadFile(const std::string filePath, std::string &content);

const std::string KBASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                         "abcdefghijklmnopqrstuvwxyz"
                                         "0123456789+/";
int Base64Decode(const std::string &enco, std::vector<unsigned char> &deco);

class TimerUtils
{
public:
    static void Start(const std::string& tag);
    static void Stop(const std::string& tag);
private:
    static inline std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
};
#endif