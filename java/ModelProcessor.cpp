#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <openssl/md5.h>
#include <exception>
#include "ModelProcessor.h"
#include "yolov11_det.hpp"
#include "tensorrt_backend.hpp"

bool isLicensed = false;

std::string calculateMD5(const std::string &machineId)
{
    unsigned char digest[MD5_DIGEST_LENGTH]; // 用于存储 MD5 结果

    // 计算 MD5 值
    MD5_CTX mdContext;
    MD5_Init(&mdContext);
    MD5_Update(&mdContext, machineId.c_str(), machineId.length());
    MD5_Final(digest, &mdContext);

    // 转为十六进制字符串
    std::ostringstream oss;
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    std::string encryptLicense = oss.str();

    return encryptLicense;
}

bool LicenseCheck(std::string key)
{
    // std::cout << "key = " << key << std::endl;
    std::ifstream ifs("/sys/class/dmi/id/product_serial");
    if (!ifs.is_open())
    {
        std::cout << "machineid file does not exist" << std::endl;
        return false;
    }
    std::string firstLine;
    if (std::getline(ifs, firstLine))
    {
        // std::cout << "first line: " << firstLine << std::endl;
        std::string license = calculateMD5(firstLine);
        // std::cout << "license = " << license << std::endl;
        // std::cout << "key = " << key << std::endl;
        if (license == key)
        {
            isLicensed = true;
            std::cout << "machine has been activated " << std::endl;
            return true;
        }
        else
        {
            std::cout << "machine does not have license " << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "machineid file is empty or failed to read" << std::endl;
    }
    ifs.close();

    return false;
}

// JNI implementations
JNIEXPORT jboolean JNICALL Java_ModelProcessor_setLicense(JNIEnv *env, jobject, jstring licenseKey)
{
    if (isLicensed)
    {
        std::cout << "Repeate license authorized " << std::endl;
        return JNI_FALSE;
    }
    const char *licenseKeyChars = env->GetStringUTFChars(licenseKey, nullptr);
    std::string licenseKeyStr(licenseKeyChars);
    env->ReleaseStringUTFChars(licenseKey, licenseKeyChars);
    if (LicenseCheck(licenseKeyStr))
    {
        return JNI_TRUE;
    }
    else
    {
        return JNI_FALSE;
    }
    // std::cout << "isLicensed = "<< isLicensed << std::endl;
}

JNIEXPORT jlong JNICALL Java_ModelProcessor_createHandler(JNIEnv *env, jobject, jstring modelPath, jlong jgpuNo)
{
    // std::cout << "createhandler" << std::endl;
    if (!isLicensed)
    {
        return reinterpret_cast<jlong>(nullptr);
    }
    const char *modelPathChars = env->GetStringUTFChars(modelPath, nullptr);
    std::string modelPathStr(modelPathChars);
    env->ReleaseStringUTFChars(modelPath, modelPathChars);
    unsigned short gpuNo = static_cast<unsigned short>(jgpuNo);
    ModelLoadOpt opt;
    opt.modelPath = modelPathStr;
    opt.deviceId = gpuNo;
    std::unique_ptr<TensorRTBackend> backend(new TensorRTBackend(opt));
    YOLOV11Det *model = new YOLOV11Det(std::move(backend));
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_ModelProcessor_destroyHandler(JNIEnv *env, jobject, jlong handlerPtr)
{
    // std::cout << "delete " << std::endl;
    if (!isLicensed || !handlerPtr)
    {
        return;
    }
    YOLOV11Det *model = reinterpret_cast<YOLOV11Det *>(handlerPtr);
    delete model;
}

JNIEXPORT jbyteArray JNICALL Java_ModelProcessor_infer(JNIEnv *env, jobject, jlong handlerPtr, jbyteArray imageData)
{
    std::ostringstream oss;
    // Get the input byte array
    jsize length = env->GetArrayLength(imageData);
    jbyte *data = env->GetByteArrayElements(imageData, nullptr);
    try
    {
        if (!isLicensed || !handlerPtr)
        {
            return 0;
        }
        YOLOV11Det *model = reinterpret_cast<YOLOV11Det *>(handlerPtr);

        std::vector<uchar> inputBuffer(data, data + length);
        cv::Mat mat = cv::imdecode(inputBuffer, cv::IMREAD_COLOR);
        if (mat.empty())
        {
            std::cerr << "Error: Failed to decode the image. empty mat" << std::endl;
            env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);
            return 0; // Return null if image decoding fails
        }
        
        auto objectVec = model->Predict(mat);
        // model->visualizeRsult(mat, objectVec);

        // Convert results to string
        std::string message;
        if (objectVec.empty())
        {
            message = "no objects detected";
        }
        else
        {
            message = "objects detected";
        }

        oss << "{\"msg\": \"" << message << "\", ";
        oss << "\"objectVec\": [";
        for (size_t i = 0; i < objectVec.size(); ++i)
        {
            oss << objectVec[i].toJson();
            if (i != objectVec.size() - 1)
            {
                oss << ", ";
            }
        }
        oss << "], \"success\": true";
        oss << "}";
    }
    catch (const std::exception &e)
    {
        oss.str("");
        oss.clear();
        oss << "{ \"msg\": \"exception occurred\", " << "\"objectVec\": [], \"success\": false }";
    }

    std::string resultStr = oss.str();

    // Release the input byte array
    env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);

    // Create a new byte array for the result
    jbyteArray result = env->NewByteArray(resultStr.size());
    env->SetByteArrayRegion(result, 0, resultStr.size(), reinterpret_cast<const jbyte *>(resultStr.c_str()));

    return result;
}
