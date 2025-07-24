package org.opencv.photo;

/* loaded from: classes2.dex */
public class TonemapMantiuk extends Tonemap {
    public TonemapMantiuk(long j) {
        super(j);
    }

    public static TonemapMantiuk __fromPtr__(long j) {
        return new TonemapMantiuk(j);
    }

    private static native void delete(long j);

    private static native float getSaturation_0(long j);

    private static native float getScale_0(long j);

    private static native void setSaturation_0(long j, float f2);

    private static native void setScale_0(long j, float f2);

    @Override // org.opencv.photo.Tonemap, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public float getSaturation() {
        return getSaturation_0(this.nativeObj);
    }

    public float getScale() {
        return getScale_0(this.nativeObj);
    }

    public void setSaturation(float f2) {
        setSaturation_0(this.nativeObj, f2);
    }

    public void setScale(float f2) {
        setScale_0(this.nativeObj, f2);
    }
}