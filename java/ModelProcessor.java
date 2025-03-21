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

    // Declare native methods
    private native boolean setLicense(String licenseKey);
    private native long createHandler(String modelPath, long gpuNo);
    private native void destroyHandler(long handlerPtr);
    private native byte[] infer(long handlerPtr, byte[] imageData);

    private long handlerPtr;

    public void create(String modelPath){
        handlerPtr = createHandler(modelPath, 0);
    }

    public void close() {
        if (handlerPtr != 0) {
            destroyHandler(handlerPtr);
            handlerPtr = 0;
        }
    }

    public String infer(byte[] imageData) {
        byte[] resultBytes = infer(handlerPtr, imageData);
        return new String(resultBytes);
    }


    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    public static void main(String[] args) {
        try {
            ModelProcessor processor = new ModelProcessor();
            String key = "01e07321ad2e9e5129c1506d69f058ee";
            String modelPath = "/root/host_map/yolov11/lib_nvidia_ryxw.so.1.1.20250319";
            boolean isLicensed = processor.setLicense(key);
            if(!isLicensed){
                System.out.println("License is not valid");
                return;
            }
            processor.create(modelPath);
            byte[] imageData = processor.loadImage("/root/host_map/image/test.jpg");
            // System.out.println("imageData length: " + imageData.length);
            String result = processor.infer(imageData);
            // processor.saveImage(result, "output.jpg");
            System.out.println("Inference Result: " + result);
            System.out.println("Inference completed.");
            processor.close();
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

    public void saveImage(byte[] imageData, String outputPath) throws IOException {
        BufferedImage image = ImageIO.read(new java.io.ByteArrayInputStream(imageData));
        Path path = Paths.get(outputPath);
        ImageIO.write(image, "jpg", Files.newOutputStream(path));
    }
}

