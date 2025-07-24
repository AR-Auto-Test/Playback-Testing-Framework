package org.opencv.features2d;

/* loaded from: classes2.dex */
public class GFTTDetector extends Feature2D {
    public GFTTDetector(long j) {
        super(j);
    }

    public static GFTTDetector __fromPtr__(long j) {
        return new GFTTDetector(j);
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2, int i3, boolean z, double d4) {
        return __fromPtr__(create_0(i, d2, d3, i2, i3, z, d4));
    }

    private static native long create_0(int i, double d2, double d3, int i2, int i3, boolean z, double d4);

    private static native long create_1(int i, double d2, double d3, int i2, int i3, boolean z);

    private static native long create_2(int i, double d2, double d3, int i2, int i3);

    private static native long create_3(int i, double d2, double d3, int i2, boolean z, double d4);

    private static native long create_4(int i, double d2, double d3, int i2, boolean z);

    private static native long create_5(int i, double d2, double d3, int i2);

    private static native long create_6(int i, double d2, double d3);

    private static native long create_7(int i, double d2);

    private static native long create_8(int i);

    private static native long create_9();

    private static native void delete(long j);

    private static native int getBlockSize_0(long j);

    private static native String getDefaultName_0(long j);

    private static native boolean getHarrisDetector_0(long j);

    private static native double getK_0(long j);

    private static native int getMaxFeatures_0(long j);

    private static native double getMinDistance_0(long j);

    private static native double getQualityLevel_0(long j);

    private static native void setBlockSize_0(long j, int i);

    private static native void setHarrisDetector_0(long j, boolean z);

    private static native void setK_0(long j, double d2);

    private static native void setMaxFeatures_0(long j, int i);

    private static native void setMinDistance_0(long j, double d2);

    private static native void setQualityLevel_0(long j, double d2);

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public int getBlockSize() {
        return getBlockSize_0(this.nativeObj);
    }

    @Override // org.opencv.features2d.Feature2D, org.opencv.core.Algorithm
    public String getDefaultName() {
        return getDefaultName_0(this.nativeObj);
    }

    public boolean getHarrisDetector() {
        return getHarrisDetector_0(this.nativeObj);
    }

    public double getK() {
        return getK_0(this.nativeObj);
    }

    public int getMaxFeatures() {
        return getMaxFeatures_0(this.nativeObj);
    }

    public double getMinDistance() {
        return getMinDistance_0(this.nativeObj);
    }

    public double getQualityLevel() {
        return getQualityLevel_0(this.nativeObj);
    }

    public void setBlockSize(int i) {
        setBlockSize_0(this.nativeObj, i);
    }

    public void setHarrisDetector(boolean z) {
        setHarrisDetector_0(this.nativeObj, z);
    }

    public void setK(double d2) {
        setK_0(this.nativeObj, d2);
    }

    public void setMaxFeatures(int i) {
        setMaxFeatures_0(this.nativeObj, i);
    }

    public void setMinDistance(double d2) {
        setMinDistance_0(this.nativeObj, d2);
    }

    public void setQualityLevel(double d2) {
        setQualityLevel_0(this.nativeObj, d2);
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2, int i3, boolean z) {
        return __fromPtr__(create_1(i, d2, d3, i2, i3, z));
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2, int i3) {
        return __fromPtr__(create_2(i, d2, d3, i2, i3));
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2, boolean z, double d4) {
        return __fromPtr__(create_3(i, d2, d3, i2, z, d4));
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2, boolean z) {
        return __fromPtr__(create_4(i, d2, d3, i2, z));
    }

    public static GFTTDetector create(int i, double d2, double d3, int i2) {
        return __fromPtr__(create_5(i, d2, d3, i2));
    }

    public static GFTTDetector create(int i, double d2, double d3) {
        return __fromPtr__(create_6(i, d2, d3));
    }

    public static GFTTDetector create(int i, double d2) {
        return __fromPtr__(create_7(i, d2));
    }

    public static GFTTDetector create(int i) {
        return __fromPtr__(create_8(i));
    }

    public static GFTTDetector create() {
        return __fromPtr__(create_9());
    }
}