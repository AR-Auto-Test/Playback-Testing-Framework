package org.opencv.video;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

/* loaded from: classes2.dex */
public class BackgroundSubtractor extends Algorithm {
    public BackgroundSubtractor(long j) {
        super(j);
    }

    public static BackgroundSubtractor __fromPtr__(long j) {
        return new BackgroundSubtractor(j);
    }

    private static native void apply_0(long j, long j2, long j3, double d2);

    private static native void apply_1(long j, long j2, long j3);

    private static native void delete(long j);

    private static native void getBackgroundImage_0(long j, long j2);

    public void apply(Mat mat, Mat mat2, double d2) {
        apply_0(this.nativeObj, mat.nativeObj, mat2.nativeObj, d2);
    }

    @Override // org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public void getBackgroundImage(Mat mat) {
        getBackgroundImage_0(this.nativeObj, mat.nativeObj);
    }

    public void apply(Mat mat, Mat mat2) {
        apply_1(this.nativeObj, mat.nativeObj, mat2.nativeObj);
    }
}