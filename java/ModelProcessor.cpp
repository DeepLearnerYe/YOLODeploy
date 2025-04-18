#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <openssl/md5.h>
#include <exception>
#include "ModelProcessor.h"
#include "tensorrt_backend.hpp"
#include "yolov11_det.hpp"
#include "yolov11_cls.hpp"
#include "yolov11_obb.hpp"
#include "utils.hpp"


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

JNIEXPORT jlong JNICALL Java_ModelProcessor_createDetectionHandler(JNIEnv *env, jobject, jstring modelPath, jlong jgpuNo)
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

JNIEXPORT void JNICALL Java_ModelProcessor_destroyDetectionHandler(JNIEnv *env, jobject, jlong handlerPtr)
{
    // std::cout << "delete " << std::endl;
    if (!isLicensed || !handlerPtr)
    {
        return;
    }
    YOLOV11Det *model = reinterpret_cast<YOLOV11Det *>(handlerPtr);
    delete model;
}

JNIEXPORT jbyteArray JNICALL Java_ModelProcessor_detectionInfer(JNIEnv *env, jobject, jlong handlerPtr, jbyteArray imageData)
{
    std::ostringstream oss;
    // Get the input byte array
    TimerUtils::Start("javaPassImage");
    jsize length = env->GetArrayLength(imageData);
    jbyte *data = env->GetByteArrayElements(imageData, nullptr);
    
    try
    {
        if (!isLicensed || !handlerPtr)
        {
            return 0;
        }
        std::vector<uchar> inputBuffer(data, data + length);
        TimerUtils::Stop("javaPassImage");
        TimerUtils::Start("decodeImage");
        cv::Mat mat = cv::imdecode(inputBuffer, cv::IMREAD_COLOR);
        if (mat.empty())
        {
            std::cerr << "Error: Failed to decode the image. empty mat" << std::endl;
            env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);
            return 0; // Return null if image decoding fails
        }
        TimerUtils::Stop("decodeImage");

        YOLOV11Det *model = reinterpret_cast<YOLOV11Det *>(handlerPtr);
        auto objectVec = model->Predict(mat);
        // model->visualizeRsult(mat, objectVec);
        
        TimerUtils::Start("resultPassJava");
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
    TimerUtils::Stop("resultPassJava");

    return result;
}

JNIEXPORT jlong JNICALL Java_ModelProcessor_createClassificationHandler(JNIEnv *env, jobject, jstring modelPath, jlong jgpuNo)
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
    YOLOV11Cls *model = new YOLOV11Cls(std::move(backend));
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_ModelProcessor_destroyClassificationHandler(JNIEnv *env, jobject, jlong handlerPtr)
{
    // std::cout << "delete " << std::endl;
    if (!isLicensed || !handlerPtr)
    {
        return;
    }
    YOLOV11Cls *model = reinterpret_cast<YOLOV11Cls *>(handlerPtr);
    delete model;
}

JNIEXPORT jbyteArray JNICALL Java_ModelProcessor_classificationInfer(JNIEnv *env, jobject, jlong handlerPtr, jbyteArray imageData, jintArray boxesArray)
{
    std::ostringstream oss;
    // Get the boxes
    jsize boxSize = env->GetArrayLength(boxesArray);
    if(boxSize % 4 != 0)
    {
        oss.str("");
        oss.clear();
        oss << "{ \"msg\": \"boxes length invalid\", " << "\"objectVec\": [], \"success\": false }";
        std::string resultStr = oss.str();
        jbyteArray result = env->NewByteArray(resultStr.size());
        env->SetByteArrayRegion(result, 0, resultStr.size(), reinterpret_cast<const jbyte *>(resultStr.c_str()));
        return result;
    }
    jint *boxes = env->GetIntArrayElements(boxesArray, nullptr);
    std::vector<cv::Rect> roiBoxes;
    std::vector<cv::Rect> reservedBoxes;
    for (int i = 0; i < boxSize; i += 4) {
        int x = boxes[i];
        int y = boxes[i + 1];
        int w = boxes[i + 2] - boxes[i];
        int h = boxes[i + 3] - boxes[i + 1];
        roiBoxes.emplace_back(x, y, w, h);
        // std::cout << "Box " << (i / 4) << ": (" << x << ", " << y << ", " << w << ", " << h << ")" << std::endl;
    }
    env->ReleaseIntArrayElements(boxesArray, boxes, JNI_ABORT);

    // Get the input byte array
    jsize length = env->GetArrayLength(imageData);
    jbyte *data = env->GetByteArrayElements(imageData, nullptr);
    try
    {
        if (!isLicensed || !handlerPtr)
        {
            return 0;
        }
        YOLOV11Cls *model = reinterpret_cast<YOLOV11Cls *>(handlerPtr);

        std::vector<uchar> inputBuffer(data, data + length);
        cv::Mat mat = cv::imdecode(inputBuffer, cv::IMREAD_COLOR);
        if (mat.empty())
        {
            std::cerr << "Error: Failed to decode the image. empty mat" << std::endl;
            env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);
            return 0; // Return null if image decoding fails
        }

        for (auto &roi:roiBoxes)
        {
            cv::Mat cropped = mat(roi).clone();
            auto objectVec = model->Predict(cropped);
            model->visualizeRsult(cropped, objectVec);
            if(objectVec[0].className == "person")
            {
                reservedBoxes.push_back(roi);
            }
        }

        // Convert results to string
        oss << "{\"msg\": \"classification done\", ";
        oss << "\"objectVec\": [";
        for (size_t i = 0; i < reservedBoxes.size(); ++i)
        {
            oss << "{\"classVec\":{"
                << "\"x0\": " << reservedBoxes[i].x 
                << ", \"y0\": " << reservedBoxes[i].y
                << ", \"x1\": " << reservedBoxes[i].x + reservedBoxes[i].width
                << ", \"y1\": " << reservedBoxes[i].y + reservedBoxes[i].height;
            if (i != reservedBoxes.size() - 1)
            {
                oss << "}}, ";
            }
            if( i == reservedBoxes.size() - 1)
            {
                oss << "}}";
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

JNIEXPORT jlong JNICALL Java_ModelProcessor_createOrientedBoundingBoxHandler(JNIEnv *env, jobject, jstring modelPath, jlong jgpuNo)
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
    YOLOV11Obb *model = new YOLOV11Obb(std::move(backend));
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_ModelProcessor_destroyOrientedBoundingBoxHandler(JNIEnv *env, jobject, jlong handlerPtr)
{
    // std::cout << "delete " << std::endl;
    if (!isLicensed || !handlerPtr)
    {
        return;
    }
    YOLOV11Obb *model = reinterpret_cast<YOLOV11Obb *>(handlerPtr);
    delete model;
}

JNIEXPORT jbyteArray JNICALL Java_ModelProcessor_orientedBoundingBoxInfer(JNIEnv *env, jobject, jlong handlerPtr, jbyteArray imageData)
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
        YOLOV11Obb *model = reinterpret_cast<YOLOV11Obb *>(handlerPtr);

        std::vector<uchar> inputBuffer(data, data + length);
        cv::Mat mat = cv::imdecode(inputBuffer, cv::IMREAD_COLOR);
        if (mat.empty())
        {
            std::cerr << "Error: Failed to decode the image. empty mat" << std::endl;
            env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);
            return 0; // Return null if image decoding fails
        }
        
        auto objectVec = model->Predict(mat);
        model->visualizeRsult(mat, objectVec);

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