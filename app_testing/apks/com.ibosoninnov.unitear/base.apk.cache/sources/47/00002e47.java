package org.opencv.photo;

/* loaded from: classes2.dex */
public class TonemapReinhard extends Tonemap {
    public TonemapReinhard(long j) {
        super(j);
    }

    public static TonemapReinhard __fromPtr__(long j) {
        return new TonemapReinhard(j);
    }

    private static native void delete(long j);

    private static native float getColorAdaptation_0(long j);

    private static native float getIntensity_0(long j);

    private static native float getLightAdaptation_0(long j);

    private static native void setColorAdaptation_0(long j, float f2);

    private static native void setIntensity_0(long j, float f2);

    private static native void setLightAdaptation_0(long j, float f2);

    @Override // org.opencv.photo.Tonemap, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public float getColorAdaptation() {
        return getColorAdaptation_0(this.nativeObj);
    }

    public float getIntensity() {
        return getIntensity_0(this.nativeObj);
    }

    public float getLightAdaptation() {
        return getLightAdaptation_0(this.nativeObj);
    }

    public void setColorAdaptation(float f2) {
        setColorAdaptation_0(this.nativeObj, f2);
    }

    public void setIntensity(float f2) {
        setIntensity_0(this.nativeObj, f2);
    }

    public void setLightAdaptation(float f2) {
        setLightAdaptation_0(this.nativeObj, f2);
    }
}