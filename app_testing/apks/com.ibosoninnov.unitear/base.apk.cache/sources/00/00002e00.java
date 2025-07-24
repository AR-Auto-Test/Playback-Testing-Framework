package org.opencv.dnn;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.utils.Converters;

/* loaded from: classes2.dex */
public class Dnn {
    public static final int DNN_BACKEND_DEFAULT = 0;
    public static final int DNN_BACKEND_HALIDE = 1;
    public static final int DNN_BACKEND_INFERENCE_ENGINE = 2;
    public static final int DNN_BACKEND_OPENCV = 3;
    public static final int DNN_TARGET_CPU = 0;
    public static final int DNN_TARGET_FPGA = 4;
    public static final int DNN_TARGET_MYRIAD = 3;
    public static final int DNN_TARGET_OPENCL = 1;
    public static final int DNN_TARGET_OPENCL_FP16 = 2;

    public static void NMSBoxes(MatOfRect2d matOfRect2d, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt, float f4, int i) {
        NMSBoxes_0(matOfRect2d.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj, f4, i);
    }

    public static void NMSBoxesRotated(MatOfRotatedRect matOfRotatedRect, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt, float f4, int i) {
        NMSBoxesRotated_0(matOfRotatedRect.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj, f4, i);
    }

    private static native void NMSBoxesRotated_0(long j, long j2, float f2, float f3, long j3, float f4, int i);

    private static native void NMSBoxesRotated_1(long j, long j2, float f2, float f3, long j3, float f4);

    private static native void NMSBoxesRotated_2(long j, long j2, float f2, float f3, long j3);

    private static native void NMSBoxes_0(long j, long j2, float f2, float f3, long j3, float f4, int i);

    private static native void NMSBoxes_1(long j, long j2, float f2, float f3, long j3, float f4);

    private static native void NMSBoxes_2(long j, long j2, float f2, float f3, long j3);

    public static Mat blobFromImage(Mat mat, double d2, Size size, Scalar scalar, boolean z, boolean z2, int i) {
        long j = mat.nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImage_0(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z, z2, i));
    }

    private static native long blobFromImage_0(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z, boolean z2, int i);

    private static native long blobFromImage_1(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z, boolean z2);

    private static native long blobFromImage_2(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z);

    private static native long blobFromImage_3(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8);

    private static native long blobFromImage_4(long j, double d2, double d3, double d4);

    private static native long blobFromImage_5(long j, double d2);

    private static native long blobFromImage_6(long j);

    public static Mat blobFromImages(List<Mat> list, double d2, Size size, Scalar scalar, boolean z, boolean z2, int i) {
        long j = Converters.vector_Mat_to_Mat(list).nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImages_0(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z, z2, i));
    }

    private static native long blobFromImages_0(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z, boolean z2, int i);

    private static native long blobFromImages_1(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z, boolean z2);

    private static native long blobFromImages_2(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8, boolean z);

    private static native long blobFromImages_3(long j, double d2, double d3, double d4, double d5, double d6, double d7, double d8);

    private static native long blobFromImages_4(long j, double d2, double d3, double d4);

    private static native long blobFromImages_5(long j, double d2);

    private static native long blobFromImages_6(long j);

    public static List<Integer> getAvailableTargets(int i) {
        return getAvailableTargets_0(i);
    }

    private static native List<Integer> getAvailableTargets_0(int i);

    public static String getInferenceEngineBackendType() {
        return getInferenceEngineBackendType_0();
    }

    private static native String getInferenceEngineBackendType_0();

    public static String getInferenceEngineVPUType() {
        return getInferenceEngineVPUType_0();
    }

    private static native String getInferenceEngineVPUType_0();

    public static void imagesFromBlob(Mat mat, List<Mat> list) {
        Mat mat2 = new Mat();
        imagesFromBlob_0(mat.nativeObj, mat2.nativeObj);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
    }

    private static native void imagesFromBlob_0(long j, long j2);

    public static Net readNet(String str, MatOfByte matOfByte, MatOfByte matOfByte2) {
        return new Net(readNet_0(str, matOfByte.nativeObj, matOfByte2.nativeObj));
    }

    public static Net readNetFromCaffe(String str, String str2) {
        return new Net(readNetFromCaffe_0(str, str2));
    }

    private static native long readNetFromCaffe_0(String str, String str2);

    private static native long readNetFromCaffe_1(String str);

    private static native long readNetFromCaffe_2(long j, long j2);

    private static native long readNetFromCaffe_3(long j);

    public static Net readNetFromDarknet(String str, String str2) {
        return new Net(readNetFromDarknet_0(str, str2));
    }

    private static native long readNetFromDarknet_0(String str, String str2);

    private static native long readNetFromDarknet_1(String str);

    private static native long readNetFromDarknet_2(long j, long j2);

    private static native long readNetFromDarknet_3(long j);

    public static Net readNetFromModelOptimizer(String str, String str2) {
        return new Net(readNetFromModelOptimizer_0(str, str2));
    }

    private static native long readNetFromModelOptimizer_0(String str, String str2);

    private static native long readNetFromModelOptimizer_1(long j, long j2);

    public static Net readNetFromONNX(String str) {
        return new Net(readNetFromONNX_0(str));
    }

    private static native long readNetFromONNX_0(String str);

    private static native long readNetFromONNX_1(long j);

    public static Net readNetFromTensorflow(String str, String str2) {
        return new Net(readNetFromTensorflow_0(str, str2));
    }

    private static native long readNetFromTensorflow_0(String str, String str2);

    private static native long readNetFromTensorflow_1(String str);

    private static native long readNetFromTensorflow_2(long j, long j2);

    private static native long readNetFromTensorflow_3(long j);

    public static Net readNetFromTorch(String str, boolean z, boolean z2) {
        return new Net(readNetFromTorch_0(str, z, z2));
    }

    private static native long readNetFromTorch_0(String str, boolean z, boolean z2);

    private static native long readNetFromTorch_1(String str, boolean z);

    private static native long readNetFromTorch_2(String str);

    private static native long readNet_0(String str, long j, long j2);

    private static native long readNet_1(String str, long j);

    private static native long readNet_2(String str, String str2, String str3);

    private static native long readNet_3(String str, String str2);

    private static native long readNet_4(String str);

    public static Mat readTensorFromONNX(String str) {
        return new Mat(readTensorFromONNX_0(str));
    }

    private static native long readTensorFromONNX_0(String str);

    public static Mat readTorchBlob(String str, boolean z) {
        return new Mat(readTorchBlob_0(str, z));
    }

    private static native long readTorchBlob_0(String str, boolean z);

    private static native long readTorchBlob_1(String str);

    public static void resetMyriadDevice() {
        resetMyriadDevice_0();
    }

    private static native void resetMyriadDevice_0();

    public static String setInferenceEngineBackendType(String str) {
        return setInferenceEngineBackendType_0(str);
    }

    private static native String setInferenceEngineBackendType_0(String str);

    public static void shrinkCaffeModel(String str, String str2, List<String> list) {
        shrinkCaffeModel_0(str, str2, list);
    }

    private static native void shrinkCaffeModel_0(String str, String str2, List<String> list);

    private static native void shrinkCaffeModel_1(String str, String str2);

    public static void writeTextGraph(String str, String str2) {
        writeTextGraph_0(str, str2);
    }

    private static native void writeTextGraph_0(String str, String str2);

    public static void NMSBoxes(MatOfRect2d matOfRect2d, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt, float f4) {
        NMSBoxes_1(matOfRect2d.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj, f4);
    }

    public static void NMSBoxesRotated(MatOfRotatedRect matOfRotatedRect, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt, float f4) {
        NMSBoxesRotated_1(matOfRotatedRect.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj, f4);
    }

    public static Mat blobFromImage(Mat mat, double d2, Size size, Scalar scalar, boolean z, boolean z2) {
        long j = mat.nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImage_1(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z, z2));
    }

    public static Net readNet(String str, MatOfByte matOfByte) {
        return new Net(readNet_1(str, matOfByte.nativeObj));
    }

    public static Net readNetFromCaffe(String str) {
        return new Net(readNetFromCaffe_1(str));
    }

    public static Net readNetFromDarknet(String str) {
        return new Net(readNetFromDarknet_1(str));
    }

    public static Net readNetFromModelOptimizer(MatOfByte matOfByte, MatOfByte matOfByte2) {
        return new Net(readNetFromModelOptimizer_1(matOfByte.nativeObj, matOfByte2.nativeObj));
    }

    public static Net readNetFromONNX(MatOfByte matOfByte) {
        return new Net(readNetFromONNX_1(matOfByte.nativeObj));
    }

    public static Net readNetFromTensorflow(String str) {
        return new Net(readNetFromTensorflow_1(str));
    }

    public static Net readNetFromTorch(String str, boolean z) {
        return new Net(readNetFromTorch_1(str, z));
    }

    public static Mat readTorchBlob(String str) {
        return new Mat(readTorchBlob_1(str));
    }

    public static void shrinkCaffeModel(String str, String str2) {
        shrinkCaffeModel_1(str, str2);
    }

    public static void NMSBoxes(MatOfRect2d matOfRect2d, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt) {
        NMSBoxes_2(matOfRect2d.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj);
    }

    public static void NMSBoxesRotated(MatOfRotatedRect matOfRotatedRect, MatOfFloat matOfFloat, float f2, float f3, MatOfInt matOfInt) {
        NMSBoxesRotated_2(matOfRotatedRect.nativeObj, matOfFloat.nativeObj, f2, f3, matOfInt.nativeObj);
    }

    public static Mat blobFromImage(Mat mat, double d2, Size size, Scalar scalar, boolean z) {
        long j = mat.nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImage_2(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z));
    }

    public static Mat blobFromImages(List<Mat> list, double d2, Size size, Scalar scalar, boolean z, boolean z2) {
        long j = Converters.vector_Mat_to_Mat(list).nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImages_1(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z, z2));
    }

    public static Net readNet(String str, String str2, String str3) {
        return new Net(readNet_2(str, str2, str3));
    }

    public static Net readNetFromCaffe(MatOfByte matOfByte, MatOfByte matOfByte2) {
        return new Net(readNetFromCaffe_2(matOfByte.nativeObj, matOfByte2.nativeObj));
    }

    public static Net readNetFromDarknet(MatOfByte matOfByte, MatOfByte matOfByte2) {
        return new Net(readNetFromDarknet_2(matOfByte.nativeObj, matOfByte2.nativeObj));
    }

    public static Net readNetFromTensorflow(MatOfByte matOfByte, MatOfByte matOfByte2) {
        return new Net(readNetFromTensorflow_2(matOfByte.nativeObj, matOfByte2.nativeObj));
    }

    public static Net readNetFromTorch(String str) {
        return new Net(readNetFromTorch_2(str));
    }

    public static Mat blobFromImage(Mat mat, double d2, Size size, Scalar scalar) {
        long j = mat.nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImage_3(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3]));
    }

    public static Net readNet(String str, String str2) {
        return new Net(readNet_3(str, str2));
    }

    public static Net readNetFromCaffe(MatOfByte matOfByte) {
        return new Net(readNetFromCaffe_3(matOfByte.nativeObj));
    }

    public static Net readNetFromDarknet(MatOfByte matOfByte) {
        return new Net(readNetFromDarknet_3(matOfByte.nativeObj));
    }

    public static Net readNetFromTensorflow(MatOfByte matOfByte) {
        return new Net(readNetFromTensorflow_3(matOfByte.nativeObj));
    }

    public static Mat blobFromImage(Mat mat, double d2, Size size) {
        return new Mat(blobFromImage_4(mat.nativeObj, d2, size.width, size.height));
    }

    public static Mat blobFromImages(List<Mat> list, double d2, Size size, Scalar scalar, boolean z) {
        long j = Converters.vector_Mat_to_Mat(list).nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImages_2(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3], z));
    }

    public static Net readNet(String str) {
        return new Net(readNet_4(str));
    }

    public static Mat blobFromImage(Mat mat, double d2) {
        return new Mat(blobFromImage_5(mat.nativeObj, d2));
    }

    public static Mat blobFromImage(Mat mat) {
        return new Mat(blobFromImage_6(mat.nativeObj));
    }

    public static Mat blobFromImages(List<Mat> list, double d2, Size size, Scalar scalar) {
        long j = Converters.vector_Mat_to_Mat(list).nativeObj;
        double d3 = size.width;
        double d4 = size.height;
        double[] dArr = scalar.val;
        return new Mat(blobFromImages_3(j, d2, d3, d4, dArr[0], dArr[1], dArr[2], dArr[3]));
    }

    public static Mat blobFromImages(List<Mat> list, double d2, Size size) {
        return new Mat(blobFromImages_4(Converters.vector_Mat_to_Mat(list).nativeObj, d2, size.width, size.height));
    }

    public static Mat blobFromImages(List<Mat> list, double d2) {
        return new Mat(blobFromImages_5(Converters.vector_Mat_to_Mat(list).nativeObj, d2));
    }

    public static Mat blobFromImages(List<Mat> list) {
        return new Mat(blobFromImages_6(Converters.vector_Mat_to_Mat(list).nativeObj));
    }
}