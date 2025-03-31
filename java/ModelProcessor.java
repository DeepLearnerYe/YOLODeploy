import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import java.io.*;
import java.nio.file.*;


public class ModelProcessor {
    static {
        System.loadLibrary("modelprocessor"); // Load the native library
    }

    private native boolean setLicense(String licenseKey);
    // detection
    private native long createDetectionHandler(String modelPath, long gpuNo);
    private native void destroyDetectionHandler(long handlerPtr);
    private native byte[] detectionInfer(long handlerPtr, byte[] imageData);
    // classification
    private native long createClassificationHandler(String modelPath, long gpuNo);
    private native void destroyClassificationHandler(long handlerPtr);
    private native byte[] classificationInfer(long handlerPtr, byte[] imageData);
    // oriented bounding boxes
    private native long createOrientedBoundingBoxHandler(String modelPath, long gpuNo);
    private native void destroyOrientedBoundingBoxHandler(long handlerPtr);
    private native byte[] orientedBoundingBoxInfer(long handlerPtr, byte[] imageData);

    private long handlerPtr;

    public void createDetection(String modelPath){
        handlerPtr = createDetectionHandler(modelPath, 0);
    }
    public void createClassification(String modelPath){
        handlerPtr = createClassificationHandler(modelPath, 0);
    }
    public void createOrientedBoundingBox(String modelPath){
        handlerPtr = createOrientedBoundingBoxHandler(modelPath, 0);
    }

    public void closeDetection() {
        if (handlerPtr != 0) {
            destroyDetectionHandler(handlerPtr);
            handlerPtr = 0;
        }
    }
    public void closeClassification() {
        if (handlerPtr != 0) {
            destroyClassificationHandler(handlerPtr);
            handlerPtr = 0;
        }
    }
    public void closeOrientedBoundingBox() {
        if (handlerPtr != 0) {
            destroyOrientedBoundingBoxHandler(handlerPtr);
            handlerPtr = 0;
        }
    }

    public String inferDetection(byte[] imageData) {
        byte[] resultBytes = detectionInfer(handlerPtr, imageData);
        return new String(resultBytes);
    }
    public String inferClassification(byte[] imageData) {
        byte[] resultBytes = classificationInfer(handlerPtr, imageData);
        return new String(resultBytes);
    }
    public String inferOrientedBoundingBox(byte[] imageData) {
        byte[] resultBytes = orientedBoundingBoxInfer(handlerPtr, imageData);
        return new String(resultBytes);
    }

    public static void main(String[] args) {
        try {
            ModelProcessor processor = new ModelProcessor();
            String key = "01e07321ad2e9e5129c1506d69f058ee";
            boolean isLicensed = processor.setLicense(key);
            if(!isLicensed){
                System.out.println("License is not valid");
                return;
            }
            //************** example for detection
            // String detModelPath = "/root/host_map/yolov11/lib_nvidia_ryxw.so.1.1.20250319";
            // processor.createDetection(detModelPath);
            // byte[] imageData = processor.loadImage("/root/host_map/image/test.jpg");
            // String result = processor.inferDetection(imageData);
            // System.out.println("Inference Result: " + result);
            // processor.closeDetection();


            //************** example for classification
            // String clsModelPath = "/root/host_map/yolov11/lib_nvidia_person_cls.so.1.1.20250326";
            // processor.createClassification(clsModelPath);
            // byte[] imageData2 = processor.loadImage("/root/host_map/image/person_cls/6.jpg");
            // String result2 = processor.inferClassification(imageData2);
            // System.out.println("Inference Result: " + result2);
            // processor.closeClassification();

            //************** example for oriented bounding boxes
            String obbModelPath = "/root/host_map/yolov11/lib_nvidia_jjd.so.1.1.20250328";
            processor.createOrientedBoundingBox(obbModelPath);
            byte[] imageData2 = processor.loadImage("/root/host_map/image/obb/jjd4.jpg");
            String result2 = processor.inferOrientedBoundingBox(imageData2);
            System.out.println("Inference Result: " + result2);
            processor.closeOrientedBoundingBox();            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Helper methods for loading and saving images
    public byte[] loadImage(String imagePath) throws IOException {
        Path path = Paths.get(imagePath);
        BufferedImage image = ImageIO.read(Files.newInputStream(path));
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", baos);
        baos.flush();
        return baos.toByteArray();
    }
}

