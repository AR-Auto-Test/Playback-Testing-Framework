package org.opencv.photo;

import org.opencv.core.Mat;

/* loaded from: classes2.dex */
public class CalibrateRobertson extends CalibrateCRF {
    public CalibrateRobertson(long j) {
        super(j);
    }

    public static CalibrateRobertson __fromPtr__(long j) {
        return new CalibrateRobertson(j);
    }

    private static native void delete(long j);

    private static native int getMaxIter_0(long j);

    private static native long getRadiance_0(long j);

    private static native float getThreshold_0(long j);

    private static native void setMaxIter_0(long j, int i);

    private static native void setThreshold_0(long j, float f2);

    @Override // org.opencv.photo.CalibrateCRF, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public int getMaxIter() {
        return getMaxIter_0(this.nativeObj);
    }

    public Mat getRadiance() {
        return new Mat(getRadiance_0(this.nativeObj));
    }

    public float getThreshold() {
        return getThreshold_0(this.nativeObj);
    }

    public void setMaxIter(int i) {
        setMaxIter_0(this.nativeObj, i);
    }

    public void setThreshold(float f2) {
        setThreshold_0(this.nativeObj, f2);
    }
}