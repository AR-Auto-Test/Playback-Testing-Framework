package org.opencv.video;

/* loaded from: classes2.dex */
public class DualTVL1OpticalFlow extends DenseOpticalFlow {
    public DualTVL1OpticalFlow(long j) {
        super(j);
    }

    public static DualTVL1OpticalFlow __fromPtr__(long j) {
        return new DualTVL1OpticalFlow(j);
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7, int i5, boolean z) {
        return __fromPtr__(create_0(d2, d3, d4, i, i2, d5, i3, i4, d6, d7, i5, z));
    }

    private static native long create_0(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7, int i5, boolean z);

    private static native long create_1(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7, int i5);

    private static native long create_10(double d2, double d3);

    private static native long create_11(double d2);

    private static native long create_12();

    private static native long create_2(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7);

    private static native long create_3(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6);

    private static native long create_4(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4);

    private static native long create_5(double d2, double d3, double d4, int i, int i2, double d5, int i3);

    private static native long create_6(double d2, double d3, double d4, int i, int i2, double d5);

    private static native long create_7(double d2, double d3, double d4, int i, int i2);

    private static native long create_8(double d2, double d3, double d4, int i);

    private static native long create_9(double d2, double d3, double d4);

    private static native void delete(long j);

    private static native double getEpsilon_0(long j);

    private static native double getGamma_0(long j);

    private static native int getInnerIterations_0(long j);

    private static native double getLambda_0(long j);

    private static native int getMedianFiltering_0(long j);

    private static native int getOuterIterations_0(long j);

    private static native double getScaleStep_0(long j);

    private static native int getScalesNumber_0(long j);

    private static native double getTau_0(long j);

    private static native double getTheta_0(long j);

    private static native boolean getUseInitialFlow_0(long j);

    private static native int getWarpingsNumber_0(long j);

    private static native void setEpsilon_0(long j, double d2);

    private static native void setGamma_0(long j, double d2);

    private static native void setInnerIterations_0(long j, int i);

    private static native void setLambda_0(long j, double d2);

    private static native void setMedianFiltering_0(long j, int i);

    private static native void setOuterIterations_0(long j, int i);

    private static native void setScaleStep_0(long j, double d2);

    private static native void setScalesNumber_0(long j, int i);

    private static native void setTau_0(long j, double d2);

    private static native void setTheta_0(long j, double d2);

    private static native void setUseInitialFlow_0(long j, boolean z);

    private static native void setWarpingsNumber_0(long j, int i);

    @Override // org.opencv.video.DenseOpticalFlow, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public double getEpsilon() {
        return getEpsilon_0(this.nativeObj);
    }

    public double getGamma() {
        return getGamma_0(this.nativeObj);
    }

    public int getInnerIterations() {
        return getInnerIterations_0(this.nativeObj);
    }

    public double getLambda() {
        return getLambda_0(this.nativeObj);
    }

    public int getMedianFiltering() {
        return getMedianFiltering_0(this.nativeObj);
    }

    public int getOuterIterations() {
        return getOuterIterations_0(this.nativeObj);
    }

    public double getScaleStep() {
        return getScaleStep_0(this.nativeObj);
    }

    public int getScalesNumber() {
        return getScalesNumber_0(this.nativeObj);
    }

    public double getTau() {
        return getTau_0(this.nativeObj);
    }

    public double getTheta() {
        return getTheta_0(this.nativeObj);
    }

    public boolean getUseInitialFlow() {
        return getUseInitialFlow_0(this.nativeObj);
    }

    public int getWarpingsNumber() {
        return getWarpingsNumber_0(this.nativeObj);
    }

    public void setEpsilon(double d2) {
        setEpsilon_0(this.nativeObj, d2);
    }

    public void setGamma(double d2) {
        setGamma_0(this.nativeObj, d2);
    }

    public void setInnerIterations(int i) {
        setInnerIterations_0(this.nativeObj, i);
    }

    public void setLambda(double d2) {
        setLambda_0(this.nativeObj, d2);
    }

    public void setMedianFiltering(int i) {
        setMedianFiltering_0(this.nativeObj, i);
    }

    public void setOuterIterations(int i) {
        setOuterIterations_0(this.nativeObj, i);
    }

    public void setScaleStep(double d2) {
        setScaleStep_0(this.nativeObj, d2);
    }

    public void setScalesNumber(int i) {
        setScalesNumber_0(this.nativeObj, i);
    }

    public void setTau(double d2) {
        setTau_0(this.nativeObj, d2);
    }

    public void setTheta(double d2) {
        setTheta_0(this.nativeObj, d2);
    }

    public void setUseInitialFlow(boolean z) {
        setUseInitialFlow_0(this.nativeObj, z);
    }

    public void setWarpingsNumber(int i) {
        setWarpingsNumber_0(this.nativeObj, i);
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7, int i5) {
        return __fromPtr__(create_1(d2, d3, d4, i, i2, d5, i3, i4, d6, d7, i5));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6, double d7) {
        return __fromPtr__(create_2(d2, d3, d4, i, i2, d5, i3, i4, d6, d7));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4, double d6) {
        return __fromPtr__(create_3(d2, d3, d4, i, i2, d5, i3, i4, d6));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3, int i4) {
        return __fromPtr__(create_4(d2, d3, d4, i, i2, d5, i3, i4));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5, int i3) {
        return __fromPtr__(create_5(d2, d3, d4, i, i2, d5, i3));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2, double d5) {
        return __fromPtr__(create_6(d2, d3, d4, i, i2, d5));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i, int i2) {
        return __fromPtr__(create_7(d2, d3, d4, i, i2));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4, int i) {
        return __fromPtr__(create_8(d2, d3, d4, i));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3, double d4) {
        return __fromPtr__(create_9(d2, d3, d4));
    }

    public static DualTVL1OpticalFlow create(double d2, double d3) {
        return __fromPtr__(create_10(d2, d3));
    }

    public static DualTVL1OpticalFlow create(double d2) {
        return __fromPtr__(create_11(d2));
    }

    public static DualTVL1OpticalFlow create() {
        return __fromPtr__(create_12());
    }
}