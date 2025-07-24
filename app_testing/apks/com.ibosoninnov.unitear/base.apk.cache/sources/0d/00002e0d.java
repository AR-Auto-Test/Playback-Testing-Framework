package org.opencv.features2d;

import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;

/* loaded from: classes2.dex */
public class BRISK extends Feature2D {
    public BRISK(long j) {
        super(j);
    }

    public static BRISK __fromPtr__(long j) {
        return new BRISK(j);
    }

    public static BRISK create(int i, int i2, MatOfFloat matOfFloat, MatOfInt matOfInt, float f2, float f3, MatOfInt matOfInt2) {
        return __fromPtr__(create_0(i, i2, matOfFloat.nativeObj, matOfInt.nativeObj, f2, f3, matOfInt2.nativeObj));
    }

    private static native long create_0(int i, int i2, long j, long j2, float f2, float f3, long j3);

    private static native long create_1(int i, int i2, long j, long j2, float f2, float f3);

    private static native long create_10(long j, long j2, float f2);

    private static native long create_11(long j, long j2);

    private static native long create_2(int i, int i2, long j, long j2, float f2);

    private static native long create_3(int i, int i2, long j, long j2);

    private static native long create_4(int i, int i2, float f2);

    private static native long create_5(int i, int i2);

    private static native long create_6(int i);

    private static native long create_7();

    private static native long create_8(long j, long j2, float f2, float f3, long j3);

    private static native long create_9(long j, long j2, float f2, float f3);

    private static native void delete(long j);

    private static native String getDefaultName_0(long j);

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public String getDefaultName() {
        return getDefaultName_0(this.nativeObj);
    }

    public static BRISK create(int i, int i2, MatOfFloat matOfFloat, MatOfInt matOfInt, float f2, float f3) {
        return __fromPtr__(create_1(i, i2, matOfFloat.nativeObj, matOfInt.nativeObj, f2, f3));
    }

    public static BRISK create(int i, int i2, MatOfFloat matOfFloat, MatOfInt matOfInt, float f2) {
        return __fromPtr__(create_2(i, i2, matOfFloat.nativeObj, matOfInt.nativeObj, f2));
    }

    public static BRISK create(int i, int i2, MatOfFloat matOfFloat, MatOfInt matOfInt) {
        return __fromPtr__(create_3(i, i2, matOfFloat.nativeObj, matOfInt.nativeObj));
    }

    public static BRISK create(int i, int i2, float f2) {
        return __fromPtr__(create_4(i, i2, f2));
    }

    public static BRISK create(int i, int i2) {
        return __fromPtr__(create_5(i, i2));
    }

    public static BRISK create(int i) {
        return __fromPtr__(create_6(i));
    }

    public static BRISK create() {
        return __fromPtr__(create_7());
    }

    public static BRISK create(MatOfFloat matOfFloat, MatOfInt matOfInt, float f2, float f3, MatOfInt matOfInt2) {
        return __fromPtr__(create_8(matOfFloat.nativeObj, matOfInt.nativeObj, f2, f3, matOfInt2.nativeObj));
    }

    public static BRISK create(MatOfFloat matOfFloat, MatOfInt matOfInt, float f2, float f3) {
        return __fromPtr__(create_9(matOfFloat.nativeObj, matOfInt.nativeObj, f2, f3));
    }

    public static BRISK create(MatOfFloat matOfFloat, MatOfInt matOfInt, float f2) {
        return __fromPtr__(create_10(matOfFloat.nativeObj, matOfInt.nativeObj, f2));
    }

    public static BRISK create(MatOfFloat matOfFloat, MatOfInt matOfInt) {
        return __fromPtr__(create_11(matOfFloat.nativeObj, matOfInt.nativeObj));
    }
}