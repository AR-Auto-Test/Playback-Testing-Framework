package org.opencv.photo;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

/* loaded from: classes2.dex */
public class Tonemap extends Algorithm {
    public Tonemap(long j) {
        super(j);
    }

    public static Tonemap __fromPtr__(long j) {
        return new Tonemap(j);
    }

    private static native void delete(long j);

    private static native float getGamma_0(long j);

    private static native void process_0(long j, long j2, long j3);

    private static native void setGamma_0(long j, float f2);

    @Override // org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public float getGamma() {
        return getGamma_0(this.nativeObj);
    }

    public void process(Mat mat, Mat mat2) {
        process_0(this.nativeObj, mat.nativeObj, mat2.nativeObj);
    }

    public void setGamma(float f2) {
        setGamma_0(this.nativeObj, f2);
    }
}