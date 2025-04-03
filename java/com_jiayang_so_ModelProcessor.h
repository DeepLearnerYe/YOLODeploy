/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_jiayang_so_ModelProcessor */

#ifndef _Included_com_jiayang_so_ModelProcessor
#define _Included_com_jiayang_so_ModelProcessor
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    setLicense
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_jiayang_so_ModelProcessor_setLicense
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    createDetectionHandler
 * Signature: (Ljava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_com_jiayang_so_ModelProcessor_createDetectionHandler
  (JNIEnv *, jobject, jstring, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    destroyDetectionHandler
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiayang_so_ModelProcessor_destroyDetectionHandler
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    detectionInfer
 * Signature: (J[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_jiayang_so_ModelProcessor_detectionInfer
  (JNIEnv *, jobject, jlong, jbyteArray);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    createClassificationHandler
 * Signature: (Ljava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_com_jiayang_so_ModelProcessor_createClassificationHandler
  (JNIEnv *, jobject, jstring, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    destroyClassificationHandler
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiayang_so_ModelProcessor_destroyClassificationHandler
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    classificationInfer
 * Signature: (J[B[F)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_jiayang_so_ModelProcessor_classificationInfer
  (JNIEnv *, jobject, jlong, jbyteArray, jintArray);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    createOrientedBoundingBoxHandler
 * Signature: (Ljava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_com_jiayang_so_ModelProcessor_createOrientedBoundingBoxHandler
  (JNIEnv *, jobject, jstring, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    destroyOrientedBoundingBoxHandler
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_jiayang_so_ModelProcessor_destroyOrientedBoundingBoxHandler
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_jiayang_so_ModelProcessor
 * Method:    orientedBoundingBoxInfer
 * Signature: (J[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_jiayang_so_ModelProcessor_orientedBoundingBoxInfer
  (JNIEnv *, jobject, jlong, jbyteArray);

#ifdef __cplusplus
}
#endif
#endif
