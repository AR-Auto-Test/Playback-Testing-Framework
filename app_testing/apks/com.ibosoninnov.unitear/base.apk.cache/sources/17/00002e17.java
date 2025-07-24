package org.opencv.features2d;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.utils.Converters;

/* loaded from: classes2.dex */
public class MSER extends Feature2D {
    public MSER(long j) {
        super(j);
    }

    public static MSER __fromPtr__(long j) {
        return new MSER(j);
    }

    public static MSER create(int i, int i2, int i3, double d2, double d3, int i4, double d4, double d5, int i5) {
        return __fromPtr__(create_0(i, i2, i3, d2, d3, i4, d4, d5, i5));
    }

    private static native long create_0(int i, int i2, int i3, double d2, double d3, int i4, double d4, double d5, int i5);

    private static native long create_1(int i, int i2, int i3, double d2, double d3, int i4, double d4, double d5);

    private static native long create_2(int i, int i2, int i3, double d2, double d3, int i4, double d4);

    private static native long create_3(int i, int i2, int i3, double d2, double d3, int i4);

    private static native long create_4(int i, int i2, int i3, double d2, double d3);

    private static native long create_5(int i, int i2, int i3, double d2);

    private static native long create_6(int i, int i2, int i3);

    private static native long create_7(int i, int i2);

    private static native long create_8(int i);

    private static native long create_9();

    private static native void delete(long j);

    private static native void detectRegions_0(long j, long j2, long j3, long j4);

    private static native String getDefaultName_0(long j);

    private static native int getDelta_0(long j);

    private static native int getMaxArea_0(long j);

    private static native int getMinArea_0(long j);

    private static native boolean getPass2Only_0(long j);

    private static native void setDelta_0(long j, int i);

    private static native void setMaxArea_0(long j, int i);

    private static native void setMinArea_0(long j, int i);

    private static native void setPass2Only_0(long j, boolean z);

    public void detectRegions(Mat mat, List<MatOfPoint> list, MatOfRect matOfRect) {
        Mat mat2 = new Mat();
        detectRegions_0(this.nativeObj, mat.nativeObj, mat2.nativeObj, matOfRect.nativeObj);
        Converters.Mat_to_vector_vector_Point(mat2, list);
        mat2.release();
    }

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public String getDefaultName() {
        return getDefaultName_0(this.nativeObj);
    }

    public int getDelta() {
        return getDelta_0(this.nativeObj);
    }

    public int getMaxArea() {
        return getMaxArea_0(this.nativeObj);
    }

    public int getMinArea() {
        return getMinArea_0(this.nativeObj);
    }

    public boolean getPass2Only() {
        return getPass2Only_0(this.nativeObj);
    }

    public void setDelta(int i) {
        setDelta_0(this.nativeObj, i);
    }

    public void setMaxArea(int i) {
        setMaxArea_0(this.nativeObj, i);
    }

    public void setMinArea(int i) {
        setMinArea_0(this.nativeObj, i);
    }

    public void setPass2Only(boolean z) {
        setPass2Only_0(this.nativeObj, z);
    }

    public static MSER create(int i, int i2, int i3, double d2, double d3, int i4, double d4, double d5) {
        return __fromPtr__(create_1(i, i2, i3, d2, d3, i4, d4, d5));
    }

    public static MSER create(int i, int i2, int i3, double d2, double d3, int i4, double d4) {
        return __fromPtr__(create_2(i, i2, i3, d2, d3, i4, d4));
    }

    public static MSER create(int i, int i2, int i3, double d2, double d3, int i4) {
        return __fromPtr__(create_3(i, i2, i3, d2, d3, i4));
    }

    public static MSER create(int i, int i2, int i3, double d2, double d3) {
        return __fromPtr__(create_4(i, i2, i3, d2, d3));
    }

    public static MSER create(int i, int i2, int i3, double d2) {
        return __fromPtr__(create_5(i, i2, i3, d2));
    }

    public static MSER create(int i, int i2, int i3) {
        return __fromPtr__(create_6(i, i2, i3));
    }

    public static MSER create(int i, int i2) {
        return __fromPtr__(create_7(i, i2));
    }

    public static MSER create(int i) {
        return __fromPtr__(create_8(i));
    }

    public static MSER create() {
        return __fromPtr__(create_9());
    }
}