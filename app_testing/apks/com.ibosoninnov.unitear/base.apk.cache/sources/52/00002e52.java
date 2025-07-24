package org.opencv.video;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.utils.Converters;

/* loaded from: classes2.dex */
public class Video {
    private static final int CV_LKFLOW_GET_MIN_EIGENVALS = 8;
    private static final int CV_LKFLOW_INITIAL_GUESSES = 4;
    public static final int MOTION_AFFINE = 2;
    public static final int MOTION_EUCLIDEAN = 1;
    public static final int MOTION_HOMOGRAPHY = 3;
    public static final int MOTION_TRANSLATION = 0;
    public static final int OPTFLOW_FARNEBACK_GAUSSIAN = 256;
    public static final int OPTFLOW_LK_GET_MIN_EIGENVALS = 8;
    public static final int OPTFLOW_USE_INITIAL_FLOW = 4;

    public static RotatedRect CamShift(Mat mat, Rect rect, TermCriteria termCriteria) {
        double[] dArr = new double[4];
        RotatedRect rotatedRect = new RotatedRect(CamShift_0(mat.nativeObj, rect.x, rect.y, rect.width, rect.height, dArr, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon));
        rect.x = (int) dArr[0];
        rect.y = (int) dArr[1];
        rect.width = (int) dArr[2];
        rect.height = (int) dArr[3];
        return rotatedRect;
    }

    private static native double[] CamShift_0(long j, int i, int i2, int i3, int i4, double[] dArr, int i5, int i6, double d2);

    public static int buildOpticalFlowPyramid(Mat mat, List<Mat> list, Size size, int i, boolean z, int i2, int i3, boolean z2) {
        Mat mat2 = new Mat();
        int buildOpticalFlowPyramid_0 = buildOpticalFlowPyramid_0(mat.nativeObj, mat2.nativeObj, size.width, size.height, i, z, i2, i3, z2);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
        return buildOpticalFlowPyramid_0;
    }

    private static native int buildOpticalFlowPyramid_0(long j, long j2, double d2, double d3, int i, boolean z, int i2, int i3, boolean z2);

    private static native int buildOpticalFlowPyramid_1(long j, long j2, double d2, double d3, int i, boolean z, int i2, int i3);

    private static native int buildOpticalFlowPyramid_2(long j, long j2, double d2, double d3, int i, boolean z, int i2);

    private static native int buildOpticalFlowPyramid_3(long j, long j2, double d2, double d3, int i, boolean z);

    private static native int buildOpticalFlowPyramid_4(long j, long j2, double d2, double d3, int i);

    public static void calcOpticalFlowFarneback(Mat mat, Mat mat2, Mat mat3, double d2, int i, int i2, int i3, int i4, double d3, int i5) {
        calcOpticalFlowFarneback_0(mat.nativeObj, mat2.nativeObj, mat3.nativeObj, d2, i, i2, i3, i4, d3, i5);
    }

    private static native void calcOpticalFlowFarneback_0(long j, long j2, long j3, double d2, int i, int i2, int i3, int i4, double d3, int i5);

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat, Size size, int i, TermCriteria termCriteria, int i2, double d2) {
        calcOpticalFlowPyrLK_0(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj, size.width, size.height, i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon, i2, d2);
    }

    private static native void calcOpticalFlowPyrLK_0(long j, long j2, long j3, long j4, long j5, long j6, double d2, double d3, int i, int i2, int i3, double d4, int i4, double d5);

    private static native void calcOpticalFlowPyrLK_1(long j, long j2, long j3, long j4, long j5, long j6, double d2, double d3, int i, int i2, int i3, double d4, int i4);

    private static native void calcOpticalFlowPyrLK_2(long j, long j2, long j3, long j4, long j5, long j6, double d2, double d3, int i, int i2, int i3, double d4);

    private static native void calcOpticalFlowPyrLK_3(long j, long j2, long j3, long j4, long j5, long j6, double d2, double d3, int i);

    private static native void calcOpticalFlowPyrLK_4(long j, long j2, long j3, long j4, long j5, long j6, double d2, double d3);

    private static native void calcOpticalFlowPyrLK_5(long j, long j2, long j3, long j4, long j5, long j6);

    public static double computeECC(Mat mat, Mat mat2, Mat mat3) {
        return computeECC_0(mat.nativeObj, mat2.nativeObj, mat3.nativeObj);
    }

    private static native double computeECC_0(long j, long j2, long j3);

    private static native double computeECC_1(long j, long j2);

    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int i, double d2, boolean z) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_0(i, d2, z));
    }

    private static native long createBackgroundSubtractorKNN_0(int i, double d2, boolean z);

    private static native long createBackgroundSubtractorKNN_1(int i, double d2);

    private static native long createBackgroundSubtractorKNN_2(int i);

    private static native long createBackgroundSubtractorKNN_3();

    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int i, double d2, boolean z) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_0(i, d2, z));
    }

    private static native long createBackgroundSubtractorMOG2_0(int i, double d2, boolean z);

    private static native long createBackgroundSubtractorMOG2_1(int i, double d2);

    private static native long createBackgroundSubtractorMOG2_2(int i);

    private static native long createBackgroundSubtractorMOG2_3();

    public static DualTVL1OpticalFlow createOptFlow_DualTVL1() {
        return DualTVL1OpticalFlow.__fromPtr__(createOptFlow_DualTVL1_0());
    }

    private static native long createOptFlow_DualTVL1_0();

    public static Mat estimateRigidTransform(Mat mat, Mat mat2, boolean z, int i, double d2, int i2) {
        return new Mat(estimateRigidTransform_0(mat.nativeObj, mat2.nativeObj, z, i, d2, i2));
    }

    private static native long estimateRigidTransform_0(long j, long j2, boolean z, int i, double d2, int i2);

    private static native long estimateRigidTransform_1(long j, long j2, boolean z);

    public static double findTransformECC(Mat mat, Mat mat2, Mat mat3, int i, TermCriteria termCriteria, Mat mat4, int i2) {
        return findTransformECC_0(mat.nativeObj, mat2.nativeObj, mat3.nativeObj, i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon, mat4.nativeObj, i2);
    }

    private static native double findTransformECC_0(long j, long j2, long j3, int i, int i2, int i3, double d2, long j4, int i4);

    public static int meanShift(Mat mat, Rect rect, TermCriteria termCriteria) {
        double[] dArr = new double[4];
        int meanShift_0 = meanShift_0(mat.nativeObj, rect.x, rect.y, rect.width, rect.height, dArr, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon);
        rect.x = (int) dArr[0];
        rect.y = (int) dArr[1];
        rect.width = (int) dArr[2];
        rect.height = (int) dArr[3];
        return meanShift_0;
    }

    private static native int meanShift_0(long j, int i, int i2, int i3, int i4, double[] dArr, int i5, int i6, double d2);

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat, Size size, int i, TermCriteria termCriteria, int i2) {
        calcOpticalFlowPyrLK_1(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj, size.width, size.height, i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon, i2);
    }

    public static double computeECC(Mat mat, Mat mat2) {
        return computeECC_1(mat.nativeObj, mat2.nativeObj);
    }

    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int i, double d2) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_1(i, d2));
    }

    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int i, double d2) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_1(i, d2));
    }

    public static Mat estimateRigidTransform(Mat mat, Mat mat2, boolean z) {
        return new Mat(estimateRigidTransform_1(mat.nativeObj, mat2.nativeObj, z));
    }

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat, Size size, int i, TermCriteria termCriteria) {
        calcOpticalFlowPyrLK_2(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj, size.width, size.height, i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon);
    }

    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN(int i) {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_2(i));
    }

    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(int i) {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_2(i));
    }

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat, Size size, int i) {
        calcOpticalFlowPyrLK_3(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj, size.width, size.height, i);
    }

    public static BackgroundSubtractorKNN createBackgroundSubtractorKNN() {
        return BackgroundSubtractorKNN.__fromPtr__(createBackgroundSubtractorKNN_3());
    }

    public static BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2() {
        return BackgroundSubtractorMOG2.__fromPtr__(createBackgroundSubtractorMOG2_3());
    }

    public static int buildOpticalFlowPyramid(Mat mat, List<Mat> list, Size size, int i, boolean z, int i2, int i3) {
        Mat mat2 = new Mat();
        int buildOpticalFlowPyramid_1 = buildOpticalFlowPyramid_1(mat.nativeObj, mat2.nativeObj, size.width, size.height, i, z, i2, i3);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
        return buildOpticalFlowPyramid_1;
    }

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat, Size size) {
        calcOpticalFlowPyrLK_4(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj, size.width, size.height);
    }

    public static void calcOpticalFlowPyrLK(Mat mat, Mat mat2, MatOfPoint2f matOfPoint2f, MatOfPoint2f matOfPoint2f2, MatOfByte matOfByte, MatOfFloat matOfFloat) {
        calcOpticalFlowPyrLK_5(mat.nativeObj, mat2.nativeObj, matOfPoint2f.nativeObj, matOfPoint2f2.nativeObj, matOfByte.nativeObj, matOfFloat.nativeObj);
    }

    public static int buildOpticalFlowPyramid(Mat mat, List<Mat> list, Size size, int i, boolean z, int i2) {
        Mat mat2 = new Mat();
        int buildOpticalFlowPyramid_2 = buildOpticalFlowPyramid_2(mat.nativeObj, mat2.nativeObj, size.width, size.height, i, z, i2);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
        return buildOpticalFlowPyramid_2;
    }

    public static int buildOpticalFlowPyramid(Mat mat, List<Mat> list, Size size, int i, boolean z) {
        Mat mat2 = new Mat();
        int buildOpticalFlowPyramid_3 = buildOpticalFlowPyramid_3(mat.nativeObj, mat2.nativeObj, size.width, size.height, i, z);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
        return buildOpticalFlowPyramid_3;
    }

    public static int buildOpticalFlowPyramid(Mat mat, List<Mat> list, Size size, int i) {
        Mat mat2 = new Mat();
        int buildOpticalFlowPyramid_4 = buildOpticalFlowPyramid_4(mat.nativeObj, mat2.nativeObj, size.width, size.height, i);
        Converters.Mat_to_vector_Mat(mat2, list);
        mat2.release();
        return buildOpticalFlowPyramid_4;
    }
}