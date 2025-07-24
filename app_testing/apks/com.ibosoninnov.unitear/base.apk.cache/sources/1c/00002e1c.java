package org.opencv.imgproc;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.Size;

/* loaded from: classes2.dex */
public class CLAHE extends Algorithm {
    public CLAHE(long j) {
        super(j);
    }

    public static CLAHE __fromPtr__(long j) {
        return new CLAHE(j);
    }

    private static native void apply_0(long j, long j2, long j3);

    private static native void collectGarbage_0(long j);

    private static native void delete(long j);

    private static native double getClipLimit_0(long j);

    private static native double[] getTilesGridSize_0(long j);

    private static native void setClipLimit_0(long j, double d2);

    private static native void setTilesGridSize_0(long j, double d2, double d3);

    public void apply(Mat mat, Mat mat2) {
        apply_0(this.nativeObj, mat.nativeObj, mat2.nativeObj);
    }

    public void collectGarbage() {
        collectGarbage_0(this.nativeObj);
    }

    @Override // org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public double getClipLimit() {
        return getClipLimit_0(this.nativeObj);
    }

    public Size getTilesGridSize() {
        return new Size(getTilesGridSize_0(this.nativeObj));
    }

    public void setClipLimit(double d2) {
        setClipLimit_0(this.nativeObj, d2);
    }

    public void setTilesGridSize(Size size) {
        setTilesGridSize_0(this.nativeObj, size.width, size.height);
    }
}